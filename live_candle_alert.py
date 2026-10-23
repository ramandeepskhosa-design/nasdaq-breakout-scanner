"""
Live candle alerts (free Yahoo Finance data) -> Telegram
===========================================================
Every 15 minutes, one minute after each candle closes, checks the LAST CLOSED
15-min candle of each market that is currently open and sends ONE Telegram
message per market listing what fired:

  IMPULSE : a single big full-body candle (body >= 80% of its range AND body
            >= 3x the average body of the previous 20 candles). No volume used.
  NOWICK  : the moment a streak reaches 3 same-direction candles whose
            closing-side wick is <= 20% of the candle's range.

Markets: NSE (Nifty 50 + Next 50 + Midcap 50), US (NASDAQ + S&P 500 list),
CRYPTO (BTC, ETH, SOL, BNB, XRP, DOGE, always open).

Usage:
  python3 live_candle_alert.py                 # run forever (needs Mac awake)
  python3 live_candle_alert.py --once          # one pass, then exit
  options: --force (ignore market hours) --dry (print, don't send)
           --only=NSE,CRYPTO (limit markets)
Env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import html
import json
import re
import os
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, ".live_alert_state.json")
LOG_FILE   = os.path.join(BASE_DIR, ".live_alert_log.jsonl")   # every alert ever sent (feeds the daily report)

IMPULSE_MIN_BODY = 80.0
IMPULSE_MULT     = 3.0
AVG_LEN          = 20
NW_MAX_WICK      = 20.0
NW_STREAK        = 3
# "Near day low/high" alert: price sits at the day's extreme after small-closing-wick candles.
EXT_NEAR_PCT      = 0.30   # close within this % (of price) of the day's low (short) / high (long)
EMA_ORDER_PERIODS = (5, 10, 20, 50, 100, 200)   # ALL six EMAs in strict order: BULL = 5>10>20>50>100>200 (ascending), BEAR = 5<10<20<50<100<200 (descending)
OLD_ALERTS = False   # 23 Oct: near-extreme and volume-surge candle alerts replaced by session_scan.py range scans; set True to bring them back
ALERT_KINDS = {"EXTREME"} if OLD_ALERTS else set()
EXT_ALLOW_TWO = False       # two-candle path switched off: a single strong candle (wick <= EXT_WICK_1) is required
VS_BODY, VS_VOL, VS_NEAR, VS_SHARE, VS_TURNOVER, VS_MIN_CANDLES = 80.0, 3.0, 1.0, 0.80, 1e7, 1   # volume-surge alert (wicks small, volume high for the time, near the day extreme, day beyond EMA5/10/20)
EXT_COOLDOWN_MIN = 60     # a stock can re-alert NEAR DAY LOW/HIGH (same direction) after this many minutes
INDEX_NAMES = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY"}
INDEX_VOLUME_PROXY = {"^NSEI": "NIFTYBEES.NS", "^NSEBANK": "BANKBEES.NS"}   # Yahoo gives zero volume for the index itself
EXT_MIN_RANGE_PCT = 0.80   # day range so far >= this % of price, else everything is "near" an extreme
EXT_MIN_BARS      = 2      # opening candle (09:15) + the signal candle (09:30 onwards): the first 15 minutes set the opening range
EXT_WICK_2        = 20.0   # path A: last 2 same-direction candles, closing-side wick <= this %
EXT_WICK_1        = 10.0   # path B: ONE strong candle, closing-side wick <= this % ...
EXT_WICK_OTHER    = 10.0   #         ... AND the opposite-side wick <= this % (a short candle must not have gone up from its open, a long must not have gone below it)
EXT_BODY_1        = 80.0   #         ... and body >= this % of its range
EXT_REQUIRE_EMA   = True   # also demand the 5/10/20/50/100/200 EMA stack (set False to drop it)
EMA_PERIODS      = (5, 10, 20, 50, 100, 200)
EMA_MIN_BARS     = 250   # EMA200 needs a long warm-up to be meaningful
BAR              = timedelta(minutes=15)
CRYPTO = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
          "BNB-USD": "BNB", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin"}


FO_FILE = os.path.join(BASE_DIR, "learning", "fo_symbols.txt")
FO_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"


def fo_symbols():
    """All NSE F&O stocks (official lot-size list). Re-downloaded at most every 3 days; the saved copy is the fallback."""
    try:
        fresh = os.path.exists(FO_FILE) and time.time() - os.path.getmtime(FO_FILE) < 3 * 86400
        if not fresh:
            r = urllib.request.Request(FO_URL, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
            text = urllib.request.urlopen(r, timeout=20).read().decode()
            skip = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI", "SYMBOL"}
            syms = []
            for line in text.splitlines()[1:]:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) > 1 and parts[1] and parts[1] not in skip and not parts[1].startswith("Symbol"):
                    syms.append(parts[1])
            if len(syms) > 150:
                os.makedirs(os.path.dirname(FO_FILE), exist_ok=True)
                open(FO_FILE, "w").write("\n".join(syms))
    except Exception as e:
        print("F&O list refresh failed, using saved copy:", str(e)[:80])
    try:
        return [x.strip() for x in open(FO_FILE).read().split() if x.strip()]
    except Exception:
        return []


def load_markets():
    uni = json.load(open(os.path.join(BASE_DIR, "universe.json")))["universe"]
    idx = json.load(open(os.path.join(BASE_DIR, "nse_index_stocks.json")))
    dead = {"TATAMOTORS", "LTIM", "ZOMATO", "MCDOWELL-N"}        # renamed/demerged: ETERNAL, LTM, TMPV, UNITDSPR are in the F&O list
    nse = [x for x in dict.fromkeys(idx["nifty50"] + idx["nifty_next50"] + idx["midcap50"] + fo_symbols()) if x not in dead]
    return {
        "NSE":    {"tickers": [f"{s}.NS" for s in nse], "tz": "Asia/Kolkata",
                   "hours": ((9, 15), (15, 35)), "first_bar": (9, 15), "last_bar": (15, 15), "report_after": (15, 31), "cur": "₹", "need_vwap": True},
        "INDEX":  {"tickers": list(INDEX_NAMES), "tz": "Asia/Kolkata", "ext_min_range": 0.35, "need_ema": False,
                   "hours": ((9, 15), (15, 35)), "first_bar": (9, 15), "last_bar": (15, 15), "report_after": (15, 31), "cur": ""},
        "US":     {"tickers": uni, "tz": "America/New_York",
                   "hours": ((9, 30), (16, 5)), "first_bar": (9, 30), "last_bar": (15, 45), "report_after": (16, 1), "cur": "$", "need_vwap": True},
        "CRYPTO": {"tickers": list(CRYPTO), "tz": "UTC", "hours": None, "report_after": (23, 46), "cur": "$"},
    }


def is_open(m, now_utc):
    if m["hours"] is None:
        return True
    local = now_utc.astimezone(ZoneInfo(m["tz"]))
    if local.weekday() >= 5:
        return False
    (h1, m1), (h2, m2) = m["hours"]
    return (h1, m1) <= (local.hour, local.minute) <= (h2, m2)


def flatten(sub):
    if isinstance(sub.columns, pd.MultiIndex):
        for lvl in range(sub.columns.nlevels):
            vals = sub.columns.get_level_values(lvl)
            if "Close" in vals:
                sub = sub.copy()
                sub.columns = vals
                break
    return sub


def fetch(tickers, chunk=50, period="2d"):
    out = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            df = yf.download(part, period=period, interval="15m", group_by="ticker",
                             threads=True, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"  fetch chunk {i} failed: {e}")
            continue
        for t in part:
            try:
                sub = df if len(part) == 1 else df[t]
                sub = flatten(sub).dropna(how="all")
                if not sub.empty:
                    out[t] = sub
            except Exception:
                continue
        time.sleep(1)
    # Yahoo rate-limits (HTTP 429) in bursts and then returns nothing for whole chunks. A few
    # genuinely delisted symbols are normal, but a large gap means an outage: back off and retry
    # the missing ones slowly (single-threaded) instead of skipping the whole 15-min candle.
    missing = [t for t in tickers if t not in out]
    for attempt in range(1, 4):
        if len(missing) <= max(5, len(tickers) // 10):
            break
        print(f"  fetch gap: {len(missing)}/{len(tickers)} missing - retry {attempt} after {20 * attempt}s")
        time.sleep(20 * attempt)
        for i in range(0, len(missing), 25):
            part = missing[i:i + 25]
            try:
                df = yf.download(part, period=period, interval="15m", group_by="ticker",
                                 threads=False, progress=False, auto_adjust=False)
            except Exception:
                continue
            for t in part:
                try:
                    sub = df if len(part) == 1 else df[t]
                    sub = flatten(sub).dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
                except Exception:
                    continue
            time.sleep(2)
        missing = [t for t in tickers if t not in out]
    return out


def ema_aligned(closes, direction=None):
    """15m EMA 5, 10, 20, 50, 100, 200 ALL in strict order that matches the alert: ascending for BULL,
    descending for BEAR. direction=None accepts either order. Evaluated at the last completed candle."""
    if len(closes) < EMA_MIN_BARS:
        return False
    e = [closes.ewm(span=p, adjust=False).mean().iloc[-1] for p in EMA_ORDER_PERIODS]
    rising = all(e[i] > e[i + 1] for i in range(len(e) - 1))
    falling = all(e[i] < e[i + 1] for i in range(len(e) - 1))
    if direction == "bull":
        return rising
    if direction == "bear":
        return falling
    return rising or falling


def vwap_at(df, bar_ts, tz):
    """Session VWAP up to and including the candle that starts at bar_ts (None when the data has no volume)."""
    try:
        d = df.dropna(subset=["High", "Low", "Close", "Volume"]).copy()
        d.index = d.index.tz_convert(tz) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(tz)
        t = pd.Timestamp(bar_ts).tz_convert(tz)
        sub = d[(d.index.date == t.date()) & (d.index <= t)]
        vol = sub["Volume"].astype(float)
        if len(sub) == 0 or vol.sum() <= 0:
            return None
        return float((((sub["High"] + sub["Low"] + sub["Close"]) / 3) * vol).sum() / vol.sum())
    except Exception:
        return None


def completed(df, now_utc):
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df[(idx + BAR) <= pd.Timestamp(now_utc)]


def evaluate(bars):
    """bars: completed 15m candles only. Returns [(kind, direction, detail)]."""
    o = bars["Open"].values.astype(float)
    h = bars["High"].values.astype(float)
    l = bars["Low"].values.astype(float)
    c = bars["Close"].values.astype(float)
    n = len(o)
    hits = []

    rng = h[-1] - l[-1]
    if rng > 0:
        body = abs(c[-1] - o[-1])
        avg = np.mean(np.abs(c[-AVG_LEN - 1:-1] - o[-AVG_LEN - 1:-1]))
        pct = body / rng * 100
        if avg > 0 and pct >= IMPULSE_MIN_BODY and body >= IMPULSE_MULT * avg:
            hits.append(("IMPULSE", "bull" if c[-1] > o[-1] else "bear",
                         f"body {pct:.0f}%, {body / avg:.1f}x normal"))

    def qualifies(i):
        r = h[i] - l[i]
        if r <= 0:
            return None
        red = c[i] < o[i]
        trailing = ((c[i] - l[i]) if red else (h[i] - c[i])) / r * 100
        return ("bear" if red else "bull") if trailing <= NW_MAX_WICK else None

    last = qualifies(n - 1)
    if last:
        run = 0
        for i in range(n - 1, -1, -1):
            if qualifies(i) == last:
                run += 1
            else:
                break
        if run == NW_STREAK:
            hits.append(("NOWICK", last, f"{run}-candle no-wick streak"))
    return hits


def evaluate_extreme(bars, tz, min_range=EXT_MIN_RANGE_PCT):
    """Near the day's LOW (short setup) or HIGH (long setup) after small-closing-wick candles.
    Path A: last 2 candles same direction, each with closing-side wick <= EXT_WICK_2.
    Path B: the last candle alone is strong: closing wick <= EXT_WICK_1 and body >= EXT_BODY_1."""
    idx = bars.index.tz_convert(tz) if bars.index.tz is not None else bars.index
    last_day = idx[-1].date()
    day = bars[np.array([ts.date() == last_day for ts in idx])]
    if len(day) < EXT_MIN_BARS:
        return []
    o, h, l, c = (day[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    px = c[-1]
    day_hi, day_lo = h.max(), l.min()
    if px <= 0 or (day_hi - day_lo) / px * 100 < min_range:
        return []

    def candle(i):
        r = h[i] - l[i]
        if r <= 0:
            return None
        red = c[i] < o[i]
        wick = ((c[i] - l[i]) if red else (h[i] - c[i])) / r * 100
        body = abs(c[i] - o[i]) / r * 100
        return ("bear" if red else "bull"), wick, body, 100.0 - wick - body

    last = candle(-1)
    if last is None:
        return []
    d, wick, body, other = last
    dist = (px - day_lo) / px * 100 if d == "bear" else (day_hi - px) / px * 100
    if dist > EXT_NEAR_PCT:
        return []
    prev = candle(-2)
    two = EXT_ALLOW_TWO and prev is not None and prev[0] == d and prev[1] <= EXT_WICK_2 and wick <= EXT_WICK_2
    one = wick <= EXT_WICK_1 and other <= EXT_WICK_OTHER and body >= EXT_BODY_1
    if not (two or one):
        return []
    colour = "red" if d == "bear" else "green"
    side = "low" if d == "bear" else "high"
    return [("EXTREME", d, f"{2 if two else 1} {colour} candle(s), closing wick {wick:.0f}%, other wick {other:.0f}%, "
                           f"{dist:.2f}% from day {side}")]


def load_state():
    try:
        return json.load(open(STATE_FILE))
    except Exception:
        return {}


def save_state(state):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    state = {k: v for k, v in state.items() if v >= cutoff}
    json.dump(state, open(STATE_FILE, "w"))


def send(text, dry):
    if dry:
        print(text)
        return
    url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"

    def post(body, parse_mode):
        fields = {"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": body}
        if parse_mode:
            fields["parse_mode"] = parse_mode
        urllib.request.urlopen(urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(), method="POST"), timeout=15)

    try:
        post(text, "HTML")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:200]
        except Exception:
            pass
        print(f"telegram HTTP {e.code}: {detail}")
        if e.code != 400:
            raise
        # 400 = Telegram could not parse the HTML (stray < or >): resend as plain text so the alert is never lost
        post(re.sub(r"</?(b|i|u|code|pre)>", "", text), None)


def send_chunks(head, lines, dry, limit=3400):
    """Send head + lines as one or more Telegram messages, each under the 4096-character limit (splits by length, not by count)."""
    chunk, size = [], len(head)
    for ln in lines:
        add = len(ln) + 1
        if chunk and size + add > limit:
            send(head + "\n".join(chunk), dry)
            chunk, size = [], len(head)
        chunk.append(ln)
        size += add
    if chunk:
        send(head + "\n".join(chunk), dry)


def append_log(entries):
    with open(LOG_FILE, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def read_log(market, day):
    seen, out = set(), []
    try:
        with open(LOG_FILE) as f:
            for line in f:
                e = json.loads(line)
                if e.get("market") != market or e.get("date") != day:
                    continue
                k = (e["t"], e["bar"], e["kind"], e["dir"])
                if k not in seen:
                    seen.add(k)
                    out.append(e)
    except FileNotFoundError:
        pass
    return out


def kind_label(kind, d):
    if kind == "VOLSURGE":
        return "VOLUME SURGE"
    if kind == "EXTREME":
        return "NEAR DAY LOW" if d == "bear" else "NEAR DAY HIGH"
    return "IMPULSE" if kind == "IMPULSE" else "NO-WICK"


def day_close_prices(m, tickers, day, now):
    """Last completed close per ticker on local date `day` (7 days of 15m history)."""
    tz = ZoneInfo(m["tz"])
    out = {}
    for t, df in fetch(sorted(tickers), period="7d").items():
        b = completed(df, now)
        if b.empty:
            continue
        idx = b.index.tz_convert(tz) if b.index.tz is not None else b.index.tz_localize("UTC").tz_convert(tz)
        sel = b[idx.strftime("%Y-%m-%d") == day]
        if len(sel):
            out[t] = float(sel["Close"].iloc[-1])
    return out


def maybe_report(name, m, now, state, data, dry):
    """Daily signal report per market after its close - with catch-up.

    Looks at today and the 3 previous days; any day that has signals (or is today) and whose
    report was never sent goes out at the next cycle after the Mac wakes, marked as late."""
    ra = m.get("report_after")
    if not ra:
        return
    local = now.astimezone(ZoneInfo(m["tz"]))
    for back in range(3, -1, -1):
        d = local - timedelta(days=back)
        if m["hours"] is not None and d.weekday() >= 5:
            continue
        if back == 0 and (local.hour, local.minute) < ra:
            continue
        day = d.strftime("%Y-%m-%d")
        key = f"REPORT|{name}|{day}"
        if key in state:
            continue
        entries = sorted(read_log(name, day), key=lambda e: e["bar"])
        if not entries and back > 0:
            continue
        due = d.replace(hour=ra[0], minute=ra[1], second=0, microsecond=0)
        late = (local - due) > timedelta(minutes=45)
        head = f"<b>📋 Daily signal report - {name} - {day}</b>" + (" <i>(late - sent after the Mac woke up)</i>" if late else "")
        if not entries:
            send(head + "\nNo signals fired today.", dry)
            state[key] = now.isoformat()
            continue
        last_px = day_close_prices(m, {e["t"] for e in entries}, day, now)
        rows = []
        for e in entries:
            lp = last_px.get(e["t"])
            mv = None
            if lp and e["px"]:
                mv = (lp - e["px"]) / e["px"] * 100
                if e["dir"] == "bear":
                    mv = -mv
            rows.append((e, lp, mv))
        kinds = {}
        for e, _, _ in rows:
            kinds[kind_label(e["kind"], e["dir"])] = kinds.get(kind_label(e["kind"], e["dir"]), 0) + 1
        moves = [mv for _, _, mv in rows if mv is not None]
        summary = [head, f"{len(rows)} signals: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))]
        if moves:
            win = sum(1 for x in moves if x > 0)
            summary.append(f"Moved the signal's way by the close: {win}/{len(moves)} ({win / len(moves) * 100:.0f}%), "
                           f"average {sum(moves) / len(moves):+.2f}%")
        lines = []
        for e, lp, mv in rows:
            icon = "🟢" if e["dir"] == "bull" else "🔴"
            tm = e["bar"][11:16]
            tail = f"→ {m['cur']}{lp:,.2f} ({mv:+.2f}%) {'✅' if mv > 0 else '❌'}" if mv is not None else ""
            lines.append(f"{icon} {tm} <b>{e['sym']}</b> {m['cur']}{e['px']:,.2f} {kind_label(e['kind'], e['dir'])} {tail}")
        for k in range(0, len(lines), 40):
            send("\n".join(summary if k == 0 else [head + " (cont.)"]) + "\n\n" + "\n".join(lines[k:k + 40]), dry)
        state[key] = now.isoformat()      # only after every part was sent


def slot_volume_ratio(df, bar_ts, tz="Asia/Kolkata"):
    """Volume of the candle starting at bar_ts vs the average of the same 15-min slot over the previous 10 sessions."""
    try:
        v = df["Volume"].copy()
        v.index = v.index.tz_convert(tz) if v.index.tz is not None else v.index.tz_localize("UTC").tz_convert(tz)
        t = pd.Timestamp(bar_ts).tz_convert(tz)
        slot = v[(v.index.hour == t.hour) & (v.index.minute == t.minute)]
        cur = slot.get(t)
        prev = slot[slot.index.date < t.date()].tail(10)
        prev = prev[prev > 0]
        if cur is None or cur <= 0 or len(prev) < 3:
            return None
        return float(cur / prev.mean())
    except Exception:
        return None


def volume_note(ratio, proxy=""):
    if ratio is None:
        return ""
    tag = "🔥" if ratio >= 2 else ("" if ratio >= 0.7 else "💤")
    return f"\n    📦 volume{proxy} {ratio:.1f}x normal for this time {tag}".rstrip()


def index_volume_note(ticker, bar_ts):
    etf = INDEX_VOLUME_PROXY.get(ticker)
    if not etf:
        return ""
    df = fetch([etf], period="15d").get(etf)
    return volume_note(slot_volume_ratio(df, bar_ts), " (ETF proxy)") if df is not None else ""


def nifty_change_from_open(now):
    """NSE only: Nifty's move since today's open at the latest completed 15m candle (None if unavailable)."""
    try:
        df = yf.download("^NSEI", period="2d", interval="15m", progress=False, auto_adjust=False)
        df = completed(flatten(df), now)
        if df.empty:
            return None
        df.index = df.index.tz_convert("Asia/Kolkata")
        day = df[df.index.strftime("%Y-%m-%d") == df.index[-1].strftime("%Y-%m-%d")]
        return float((day["Close"].iloc[-1] / day["Open"].iloc[0] - 1) * 100)
    except Exception:
        return None


def learning_note(kind, direction, bar_ts, nchg):
    """Nifty context + a star when the alert matches a setup the nightly study has PROVEN."""
    if nchg is None:
        return ""
    s = 1 if direction == "bull" else -1
    nd = nchg * s
    note = f"\n    📊 Nifty {nchg:+.2f}% today ({'with you' if nd > 0.1 else 'against you' if nd < -0.1 else 'flat'})"
    try:
        card = json.load(open(os.path.join(BASE_DIR, "learning", "scorecard.json")))
        t = pd.Timestamp(bar_ts)
        mday = t.hour * 60 + t.minute
        for r in card.get("proven", []):
            lo, hi = r["slot_range"]
            if (r["kind"] in ("ALL", kind) and r["dir"] in ("both", direction) and lo <= mday < hi
                    and (r["nifty_min"] is None or nd > r["nifty_min"])):
                note += (f"\n    ⭐ PROVEN setup: {r['kind']}/{r['dir']}/{r['slot']}/{r['nifty']} "
                         f"({r['n']} trades, {r['win']}% won, {r['ret']:+.2f}%/trade)")
                break
    except Exception:
        pass
    return note


_NSE_SESSION = None
OI_LOG = os.path.join(BASE_DIR, "learning", "oi_log.csv")


def _nse_session():
    """NSE needs a cookie warm-up before its JSON APIs answer (same trick as oi_snapshot.py)."""
    global _NSE_SESSION
    if _NSE_SESSION is None:
        import requests
        sess = requests.Session()
        sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                             "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/option-chain"})
        sess.get("https://www.nseindia.com", timeout=8)
        sess.get("https://www.nseindia.com/option-chain", timeout=8)
        _NSE_SESSION = sess
    return _NSE_SESSION


def oi_context(now_ist):
    """Futures+options open interest for all F&O stocks from NSE (one request): {symbol: {oi, prev, chg, pct, d15}}.
    d15 = OI change since the snapshot ~15 minutes ago (from learning/oi_log.csv), None if no snapshot. Never raises."""
    try:
        sess = _nse_session()
        j = sess.get("https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings", timeout=10).json()
        rows = j.get("data", [])
        if not rows:
            return {}
        ts = now_ist.strftime("%Y-%m-%d %H:%M")
        old = pd.DataFrame()
        try:
            old = pd.read_csv(OI_LOG, usecols=["ts", "symbol", "latestOI"])
            old["t"] = pd.to_datetime(old.ts)
            cut = pd.Timestamp(now_ist.replace(tzinfo=None))
            old = old[(old.t >= cut - pd.Timedelta(minutes=40)) & (old.t <= cut - pd.Timedelta(minutes=9))]
        except Exception:
            pass
        out, newrows = {}, []
        target = pd.Timestamp(now_ist.replace(tzinfo=None)) - pd.Timedelta(minutes=15)
        for r in rows:
            sym = r["symbol"]
            d15 = None
            if len(old):
                o = old[old.symbol == sym]
                if len(o):
                    k = (o.t - target).abs().idxmin()
                    d15 = (float(r["latestOI"]) - float(o.loc[k, "latestOI"]), int((pd.Timestamp(now_ist.replace(tzinfo=None)) - o.loc[k, "t"]).total_seconds() / 60))
            out[sym] = dict(oi=r["latestOI"], prev=r["prevOI"], chg=r["changeInOI"], pct=r["avgInOI"], d15=d15)
            newrows.append(f"{ts},{sym},{r['latestOI']},{r['prevOI']},{r['changeInOI']},{r['avgInOI']},{r['volume']},{r['underlyingValue']}\n")
        try:   # keep the OI history going from the alert cycle too (oi_snapshot.py logs it every 15 min as well)
            last = open(OI_LOG).readlines()[-1].split(",")[0] if os.path.exists(OI_LOG) else ""
            if not last or (pd.Timestamp(ts) - pd.Timestamp(last)).total_seconds() > 420:
                with open(OI_LOG, "a") as f:
                    f.writelines(newrows)
        except Exception:
            pass
        return out
    except Exception as e:
        print("OI lookup failed:", str(e)[:80])
        return {}


def oi_note(sym, direction, day_price_chg, ctx):
    """One line: OI today (and the last ~15 min) and what it usually means for this price direction."""
    c = ctx.get(sym)
    if not c:
        return ""
    pct = float(c["pct"])
    if abs(pct) < 0.3:
        meaning, ok = "OI flat", None
    elif pct > 0:
        meaning = "new shorts (short build-up)" if day_price_chg < 0 else "new longs (long build-up)"
        ok = (direction == "bear") == (day_price_chg < 0)
    else:
        meaning = "long unwinding" if day_price_chg < 0 else "short covering"
        ok = (direction == "bear") == (day_price_chg < 0)
    mark = "" if ok is None else (" \u2705 supports" if ok else " \u26a0\ufe0f against")
    line = f"\n    \U0001f4ca OI today {c['chg']:+,.0f} ({pct:+.1f}%) = {meaning}{mark}"
    if c.get("d15"):
        d, mins = c["d15"]
        line += f"; last {mins} min {d:+,.0f} ({'SHEDDING' if d < 0 else 'building' if d > 0 else 'flat'})"
    return line


def order_book_note(ticker, direction):
    """Crypto only: live Binance order-book imbalance within 0.2% of price (free, no key)."""
    try:
        sym = ticker.replace("-USD", "USDT")
        url = f"https://api.binance.com/api/v3/depth?symbol={sym}&limit=100"
        with urllib.request.urlopen(url, timeout=8) as r:
            ob = json.load(r)
        bids = [(float(a), float(b)) for a, b in ob["bids"]]
        asks = [(float(a), float(b)) for a, b in ob["asks"]]
        mid = (bids[0][0] + asks[0][0]) / 2
        bid_v = sum(a * q for a, q in bids if a >= mid * 0.998)
        ask_v = sum(a * q for a, q in asks if a <= mid * 1.002)
        tot = bid_v + ask_v
        if tot <= 0:
            return ""
        bid_pct = bid_v / tot * 100
        spread = (asks[0][0] - bids[0][0]) / mid * 100
        agrees = (bid_pct >= 55) if direction == "bull" else (bid_pct <= 45)
        mark = "✅ supports" if agrees else ("⚠️ against" if abs(bid_pct - 50) >= 5 else "neutral")
        return f"\n    📖 book: bids {bid_pct:.0f}% / asks {100 - bid_pct:.0f}% within 0.2%, spread {spread:.3f}% ({mark})"
    except Exception:
        return ""


def volume_surge_alerts(name, m, data, now, state, dry):
    """Second alert type (NSE): a 15m candle with body >= VS_BODY% (small wicks) on volume >= VS_VOL x the typical volume of that
    time of day, within VS_NEAR% of the day's low (red) / high (green), in a stock that spent >= VS_SHARE of today's candles beyond
    EMA5/10/20 on the 15m chart. The 5-minute bars of the candle are checked for where the volume came in. Sent after the 15m close."""
    tz = ZoneInfo(m["tz"])
    cands = []
    for t, df in data.items():
        b = completed(df, now)
        if len(b) < 10:
            continue
        loc = b.index.tz_convert(tz)
        today = b[loc.strftime("%Y-%m-%d") == loc[-1].strftime("%Y-%m-%d")]
        hm = loc[-1].strftime("%H:%M")
        if len(today) < VS_MIN_CANDLES or hm == f"{m['last_bar'][0]:02d}:{m['last_bar'][1]:02d}":
            continue        # alerts start with the 09:15 candle (sent ~09:31); only the closing-drift candle is skipped
        r = today.iloc[-1]
        rng = float(r.High - r.Low)
        if rng <= 0 or abs(r.Close - r.Open) / rng * 100 < VS_BODY or float(r.Volume) * float(r.Close) < VS_TURNOVER:
            continue
        d = "bear" if r.Close < r.Open else "bull"
        near = (r.Close / today.Low.min() - 1) * 100 if d == "bear" else (today.High.max() / r.Close - 1) * 100
        if near > VS_NEAR:
            continue
        prev_slot = b[(loc.strftime("%H:%M") == hm) & (loc.strftime("%Y-%m-%d") < loc[-1].strftime("%Y-%m-%d"))].Volume.tail(1)
        if len(prev_slot) and prev_slot.iloc[0] > 0 and float(r.Volume) / float(prev_slot.iloc[0]) < 1.5:
            continue
        cands.append((t, d, near, float(r.Volume) * float(r.Close)))
    if not cands:
        return 0
    cands = sorted(cands, key=lambda x: -x[3])[:40]
    global oi_ctx_global
    if not oi_ctx_global and name == "NSE":
        oi_ctx_global = oi_context(now.astimezone(tz))
    long_data = fetch([c[0] for c in cands], period="30d")
    nchg = nifty_change_from_open(now) if name == "NSE" else None
    five = {}
    try:
        import yfinance as _yf
        f5 = _yf.download([c[0] for c in cands], period="5d", interval="5m", group_by="ticker", progress=False, auto_adjust=False, threads=False)
        for c in cands:
            try:
                five[c[0]] = flatten(f5[c[0]] if len(cands) > 1 else f5).dropna(subset=["Close"])
            except Exception:
                pass
    except Exception:
        pass
    out = []
    for t, d, near, _ in cands:
        ldf = long_data.get(t)
        if ldf is None:
            continue
        b = completed(ldf, now)
        loc = b.index.tz_convert(tz)
        day = loc[-1].strftime("%Y-%m-%d")
        bar_ts = str(b.index[-1])
        today = b[loc.strftime("%Y-%m-%d") == day]
        e5, e10, e20 = (b.Close.ewm(span=p, adjust=False).mean() for p in (5, 10, 20))
        tl = loc.strftime("%Y-%m-%d") == day
        beyond = ((b.Close < e5) & (b.Close < e10) & (b.Close < e20)) if d == "bear" else ((b.Close > e5) & (b.Close > e10) & (b.Close > e20))
        ntoday = int(tl.sum())
        share = float(beyond[tl].mean()) if ntoday >= 6 else float(beyond.iloc[-12:].mean())   # early in the day: last 12 candles (incl. yesterday's close)
        vr = slot_volume_ratio(ldf, bar_ts, m["tz"])
        if share < VS_SHARE or vr is None or vr < VS_VOL:
            continue
        key = f"{t}|{day}|VOLSURGE|{d}"
        if key in state:
            try:
                if (now - datetime.fromisoformat(state[key])).total_seconds() < EXT_COOLDOWN_MIN * 60:
                    continue
            except Exception:
                continue
        r = today.iloc[-1]
        rng = float(r.High - r.Low)
        up = (r.High - max(r.Open, r.Close)) / rng * 100
        lo = (min(r.Open, r.Close) - r.Low) / rng * 100
        sym = t.replace(".NS", "")
        five_note = ""
        f = five.get(t)
        if f is not None and len(f):
            f = f.copy()
            f.index = f.index.tz_convert(tz)
            t0 = b.index[-1].tz_convert(tz)
            w = f[(f.index >= t0) & (f.index < t0 + timedelta(minutes=15))]
            if len(w):
                hist = f[f.index.strftime("%Y-%m-%d") < day]
                typ = hist.groupby(hist.index.strftime("%H:%M")).Volume.mean()
                best, bx = None, 0
                for ts5, r5 in w.iterrows():
                    base = typ.get(ts5.strftime("%H:%M"), 0)
                    x = float(r5.Volume) / base if base > 0 else 0
                    if x > bx:
                        best, bx = (ts5, r5), x
                if best is not None:
                    ts5, r5 = best
                    share5 = float(r5.Volume) / max(float(w.Volume.sum()), 1) * 100
                    five_note = f"\n    5m chart: peak bar {ts5.strftime('%H:%M')} {'red' if r5.Close < r5.Open else 'green'} on {bx:.1f}x typical for that 5 minutes ({share5:.0f}% of the candle's volume)"
        vw = vwap_at(ldf, bar_ts, m["tz"]) if m.get("need_vwap") else None
        vnote = f" | {'below' if r.Close < vw else 'above'} VWAP" if vw else ""
        dchg = 0.0
        try:
            dchg = (float(r.Close) / float(b.Close[loc.strftime("%Y-%m-%d") < day].iloc[-1]) - 1) * 100
        except Exception:
            pass
        oi = oi_note(sym, d, dchg, oi_ctx_global) if oi_ctx_global else ""
        nn = f" | Nifty {nchg:+.2f}% today" if nchg is not None else ""
        icon = "\U0001f534" if d == "bear" else "\U0001f7e2"
        out.append((f"{icon} <b>{sym}</b> {m['cur']}{float(r.Close):,.2f}  VOLUME SURGE {'SHORT side' if d == 'bear' else 'LONG side'} - 15m {loc[-1].strftime('%H:%M')} body {abs(r.Close - r.Open) / rng * 100:.0f}% "
                    f"(wicks {up:.0f}/{lo:.0f}%), volume {vr:.1f}x typical for this time (Rs {float(r.Volume) * float(r.Close) / 1e7:.1f} cr), {near:.2f}% from day {'low' if d == 'bear' else 'high'}, "
                    f"{share * 100:.0f}% of {'today' + chr(39) + 's candles' if ntoday >= 6 else 'the last 12 candles'} {'below' if d == 'bear' else 'above'} EMA5/10/20{vnote}{nn}"
                    f"{' | EARLY SESSION: before 10:45 these faded in the 59-day test (1h -0.14% to -0.16%)' if loc[-1].strftime('%H:%M') < '10:45' else ''}{five_note}{oi}",
                    key, dict(ts=now.isoformat(), market=name, date=day, bar=bar_ts, t=t, sym=sym, kind="VOLSURGE", dir=d, px=float(r.Close))))
    if not out:
        return 0
    head = "<b>\U0001f50a Volume-surge alert - NSE (from 09:31: 15m body 80%+, volume 3x+ typical for the time, near day low/high, mostly beyond EMA5/10/20)</b>\n"
    try:
        send_chunks(head, [x[0] for x in out], dry)
        if not dry:
            append_log([x[2] for x in out])
            for x in out:
                state[x[1]] = now.isoformat()
    except Exception as e:
        print("VOLUME SURGE SEND FAILED:", e)
    return len(out)


oi_ctx_global = {}


def run_cycle(markets, state, force, dry, only):
    now = datetime.now(timezone.utc)
    for name, m in markets.items():
        if only and name not in only:
            continue
        if not force and not is_open(m, now):
            if not dry:
                maybe_report(name, m, now, state, None, dry)   # closed market: still catch up on unsent reports
            continue
        t0 = time.time()
        data = fetch(m["tickers"])
        pending, new_keys = [], []
        for t, df in data.items():
            full = completed(df, now)
            # Look at the newest completed candle AND the one before it (if it closed < 35 min ago), so a cycle that was
            # killed, restarted or starved by Yahoo rate-limiting never loses a candle. Already-alerted signals are
            # skipped by the state keys / the 60-minute cool-down below.
            for back in (1, 0):
                bars = full.iloc[:len(full) - back]
                if len(bars) < AVG_LEN + 2:
                    continue
                if back and (now - (bars.index[-1] + BAR).to_pydatetime()) > timedelta(minutes=35):
                    continue
                ts = str(bars.index[-1])
                local_bar = bars.index[-1].astimezone(ZoneInfo(m["tz"]))
                if m.get("first_bar") and (local_bar.hour, local_bar.minute) == m["first_bar"]:
                    continue   # opening candle: gap/auction noise, never alerted
                if m.get("last_bar") and (local_bar.hour, local_bar.minute) == m["last_bar"]:
                    continue   # closing-drift candle: full body on thin volume, mostly noise
                day_local = local_bar.strftime("%Y-%m-%d")
                for kind, d, detail in evaluate(bars) + evaluate_extreme(bars, m["tz"], m.get("ext_min_range", EXT_MIN_RANGE_PCT)):
                    if kind not in ALERT_KINDS:
                        continue
                    key = f"{t}|{day_local}|EXTREME|{d}" if kind == "EXTREME" else f"{t}|{ts}|{kind}"
                    if key in state:
                        if kind != "EXTREME":
                            continue
                        try:
                            age = (now - datetime.fromisoformat(state[key])).total_seconds()
                        except Exception:
                            age = 0
                        if age < EXT_COOLDOWN_MIN * 60:
                            continue       # same stock/direction alerted less than an hour ago
                    state[key] = now.isoformat()
                    new_keys.append(key)
                    pending.append({"t": t, "kind": kind, "dir": d, "detail": detail,
                                    "px": float(bars["Close"].iloc[-1]), "bar": ts, "date": day_local})

        # Stage 2: only candidates get the long history needed for EMA5..EMA200
        lines, sent, dropped = [], [], 0
        nchg = nifty_change_from_open(now) if (pending and name == "NSE") else None
        oi_ctx = oi_context(now.astimezone(ZoneInfo("Asia/Kolkata"))) if (pending and name == "NSE") else {}
        if pending:
            long_data = fetch(sorted({p["t"] for p in pending}),
                              period="10d" if name == "CRYPTO" else "30d")
            for p in pending:
                need_ema = m.get("need_ema", True) and not (p["kind"] == "EXTREME" and not EXT_REQUIRE_EMA)
                ldf = long_data.get(p["t"])
                closes = completed(ldf, now)["Close"] if ldf is not None else None
                if need_ema and (closes is None or not ema_aligned(closes, p["dir"])):
                    dropped += 1
                    continue
                vw = vwap_at(ldf, p["bar"], m["tz"]) if (ldf is not None and m.get("need_vwap")) else None
                if vw is not None and ((p["dir"] == "bull" and p["px"] < vw) or (p["dir"] == "bear" and p["px"] > vw)):
                    dropped += 1       # price is on the wrong side of VWAP for this direction
                    continue
                sym = {**CRYPTO, **INDEX_NAMES}.get(p["t"], p["t"].replace(".NS", ""))
                book = order_book_note(p["t"], p["dir"]) if name == "CRYPTO" else ""
                if name == "NSE":
                    dchg = 0.0
                    try:
                        dd = ldf.dropna(subset=["Close"]).copy()
                        dd.index = dd.index.tz_convert("Asia/Kolkata")
                        dsr = dd.index.strftime("%Y-%m-%d")
                        dchg = (p["px"] / float(dd.Close[dsr < p["date"]].iloc[-1]) - 1) * 100
                    except Exception:
                        pass
                    book = (learning_note(p["kind"], p["dir"], p["bar"], nchg) + volume_note(slot_volume_ratio(ldf, p["bar"]) if ldf is not None else None)
                            + oi_note(p["t"].replace(".NS", ""), p["dir"], dchg, oi_ctx))
                elif name == "INDEX":
                    ok = closes is not None and ema_aligned(closes, p["dir"])
                    book = (f"\n    📈 EMA 5-200 stack: {'✅ aligned' if ok else '⚠️ not fully aligned'} (shown anyway for indices)"
                            + index_volume_note(p["t"], p["bar"]))
                if vw is not None:
                    book += f"\n    📍 VWAP {m['cur']}{vw:,.2f} - price {(p['px'] / vw - 1) * 100:+.2f}% {'above' if p['px'] >= vw else 'below'} (with you)"
                icon = "🟢" if p["dir"] == "bull" else "🔴"
                if p["kind"] == "EXTREME":
                    action = "SHORT setup" if p["dir"] == "bear" else "LONG setup"
                    lines.append(f"{icon} <b>{sym}</b> {m['cur']}{p['px']:,.2f}  {kind_label('EXTREME', p['dir'])} - {action} - {p['detail']}{book}")
                else:
                    lines.append(f"{icon} <b>{sym}</b> {m['cur']}{p['px']:,.2f}  {kind_label(p['kind'], p['dir'])} {p['dir'].upper()} - {p['detail']}{book}")
                sent.append({"ts": now.isoformat(), "market": name, "date": p["date"], "bar": p["bar"],
                             "t": p["t"], "sym": sym, "kind": p["kind"], "dir": p["dir"], "px": p["px"]})
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {len(data)} fetched, "
              f"{len(pending)} candidates, {dropped} dropped (EMA/VWAP filter), "
              f"{len(lines)} alerts, {time.time() - t0:.0f}s")
        if lines:
            try:
                head = f"<b>🕯️ Live candle alert - {name} (15m, all EMAs 5/10/20/50/100/200 ascending bull / descending bear, VWAP side)</b>\n"
                send_chunks(head, lines, dry)
                if not dry:
                    append_log(sent)
            except Exception as e:
                # delivery failed: forget these signals so the next cycle's catch-up window sends them
                print(f"ALERT SEND FAILED for {name}: {e}")
                for key in new_keys:
                    state.pop(key, None)
        if name == "NSE":
            try:
                global oi_ctx_global
                oi_ctx_global = oi_ctx if pending else {}
                if OLD_ALERTS:
                    n_vs = volume_surge_alerts(name, m, data, now, state, dry)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {n_vs} volume-surge alerts")
                import session_scan
                n_rs = session_scan.scan(sys.modules[__name__], data, now, state, dry)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {n_rs} range-scan alerts")
            except Exception as e:
                print("volume-surge check failed:", str(e)[:120])
        if not dry:
            maybe_report(name, m, now, state, data, dry)


def sleep_to_next_bar():
    """Wait for the next 15-min boundary (+60 s) using the wall clock.

    time.sleep() does not count time the Mac spends asleep, so a single long sleep
    overshoots after a wake-up; polling the real clock every 5 s fixes that."""
    now = datetime.now(timezone.utc)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=15 - now.minute % 15)
    target = nxt + timedelta(seconds=60)
    while datetime.now(timezone.utc) < target:
        time.sleep(5)


def raise_fd_limit(target=10240):
    """launchd gives agents only 256 open files; yfinance needs far more per cycle."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception as e:
        print("could not raise open-file limit:", e)


if __name__ == "__main__":
    raise_fd_limit()
    args = sys.argv[1:]
    only = next((a.split("=", 1)[1].split(",") for a in args if a.startswith("--only=")), None)
    markets, state = load_markets(), load_state()
    if "--once" in args:
        run_cycle(markets, state, "--force" in args, "--dry" in args, only)
        save_state(state)
    else:
        # Each 15-min check runs in its own short-lived process: yfinance leaks file
        # handles, so a single long-lived process ends up with "Too many open files"
        # (this killed the first version). A timeout also stops a hung fetch from
        # stalling the loop.
        print("Live candle alerts running (Ctrl+C to stop)...")
        script = os.path.abspath(__file__)
        while True:
            sleep_to_next_bar()
            try:
                subprocess.run([sys.executable, "-u", script, "--once", *args], timeout=840)
            except subprocess.TimeoutExpired:
                print("cycle timed out after 14 min - killed, continuing with next bar")
            except Exception as e:
                print("cycle error:", e)
