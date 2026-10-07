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
import json
import os
import resource
import subprocess
import sys
import time
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
EXT_MIN_RANGE_PCT = 0.80   # day range so far >= this % of price, else everything is "near" an extreme
EXT_MIN_BARS      = 4      # need ~an hour of session before day low/high means anything
EXT_WICK_2        = 20.0   # path A: last 2 same-direction candles, closing-side wick <= this %
EXT_WICK_1        = 15.0   # path B: ONE strong candle, closing-side wick <= this % ...
EXT_BODY_1        = 70.0   #         ... and body >= this % of its range
EXT_REQUIRE_EMA   = True   # also demand the 5/10/20/50/100/200 EMA stack (set False to drop it)
EMA_PERIODS      = (5, 10, 20, 50, 100, 200)
EMA_MIN_BARS     = 250   # EMA200 needs a long warm-up to be meaningful
BAR              = timedelta(minutes=15)
CRYPTO = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
          "BNB-USD": "BNB", "XRP-USD": "XRP", "DOGE-USD": "Dogecoin"}


def load_markets():
    uni = json.load(open(os.path.join(BASE_DIR, "universe.json")))["universe"]
    idx = json.load(open(os.path.join(BASE_DIR, "nse_index_stocks.json")))
    nse = list(dict.fromkeys(idx["nifty50"] + idx["nifty_next50"] + idx["midcap50"]))
    return {
        "NSE":    {"tickers": [f"{s}.NS" for s in nse], "tz": "Asia/Kolkata",
                   "hours": ((9, 15), (15, 35)), "last_bar": (15, 15), "report_after": (15, 31), "cur": "₹"},
        "US":     {"tickers": uni, "tz": "America/New_York",
                   "hours": ((9, 30), (16, 5)), "last_bar": (15, 45), "report_after": (16, 1), "cur": "$"},
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
    return out


def ema_aligned(closes, direction):
    """Bull: EMA5 > EMA10 > EMA20 > EMA50 > EMA100 > EMA200 on the 15m chart.
    Bear: the exact reverse. Evaluated at the last completed candle."""
    if len(closes) < EMA_MIN_BARS:
        return False
    e = [closes.ewm(span=p, adjust=False).mean().iloc[-1] for p in EMA_PERIODS]
    if direction == "bull":
        return all(e[i] > e[i + 1] for i in range(len(e) - 1))
    return all(e[i] < e[i + 1] for i in range(len(e) - 1))


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


def evaluate_extreme(bars, tz):
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
    if px <= 0 or (day_hi - day_lo) / px * 100 < EXT_MIN_RANGE_PCT:
        return []

    def candle(i):
        r = h[i] - l[i]
        if r <= 0:
            return None
        red = c[i] < o[i]
        wick = ((c[i] - l[i]) if red else (h[i] - c[i])) / r * 100
        return ("bear" if red else "bull"), wick, abs(c[i] - o[i]) / r * 100

    last = candle(-1)
    if last is None:
        return []
    d, wick, body = last
    dist = (px - day_lo) / px * 100 if d == "bear" else (day_hi - px) / px * 100
    if dist > EXT_NEAR_PCT:
        return []
    prev = candle(-2)
    two = prev is not None and prev[0] == d and prev[1] <= EXT_WICK_2 and wick <= EXT_WICK_2
    one = wick <= EXT_WICK_1 and body >= EXT_BODY_1
    if not (two or one):
        return []
    colour = "red" if d == "bear" else "green"
    side = "low" if d == "bear" else "high"
    return [("EXTREME", d, f"{2 if two else 1} {colour} candle(s), closing wick {wick:.0f}%, "
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
    data = urllib.parse.urlencode({"chat_id": os.environ["TELEGRAM_CHAT_ID"],
                                   "text": text, "parse_mode": "HTML"}).encode()
    urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=15)


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
    if kind == "EXTREME":
        return "NEAR DAY LOW" if d == "bear" else "NEAR DAY HIGH"
    return "IMPULSE" if kind == "IMPULSE" else "NO-WICK"


def maybe_report(name, m, now, state, data, dry):
    """Once per day, right after the market closes: every signal fired today and how it did."""
    ra = m.get("report_after")
    if not ra:
        return
    local = now.astimezone(ZoneInfo(m["tz"]))
    if m["hours"] is not None and local.weekday() >= 5:
        return
    if (local.hour, local.minute) < ra:
        return
    day = local.strftime("%Y-%m-%d")
    key = f"REPORT|{name}|{day}"
    if key in state:
        return
    state[key] = now.isoformat()
    entries = sorted(read_log(name, day), key=lambda e: e["bar"])
    last_px = {}
    for t, df in data.items():
        b = completed(df, now)
        if len(b):
            last_px[t] = float(b["Close"].iloc[-1])
    rows = []
    for e in entries:
        lp = last_px.get(e["t"])
        mv = None
        if lp and e["px"]:
            mv = (lp - e["px"]) / e["px"] * 100
            if e["dir"] == "bear":
                mv = -mv
        rows.append((e, lp, mv))
    head = f"<b>📋 Daily signal report - {name} - {day}</b>"
    if not rows:
        send(head + "\nNo signals fired today.", dry)
        return
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


def run_cycle(markets, state, force, dry, only):
    now = datetime.now(timezone.utc)
    for name, m in markets.items():
        if only and name not in only:
            continue
        if not force and not is_open(m, now):
            continue
        t0 = time.time()
        data = fetch(m["tickers"])
        pending = []
        for t, df in data.items():
            bars = completed(df, now)
            if len(bars) < AVG_LEN + 2:
                continue
            ts = str(bars.index[-1])
            local_bar = bars.index[-1].astimezone(ZoneInfo(m["tz"]))
            if m.get("last_bar") and (local_bar.hour, local_bar.minute) == m["last_bar"]:
                continue   # closing-drift candle: full body on thin volume, mostly noise
            day_local = local_bar.strftime("%Y-%m-%d")
            for kind, d, detail in evaluate(bars) + evaluate_extreme(bars, m["tz"]):
                key = f"{t}|{day_local}|EXTREME|{d}" if kind == "EXTREME" else f"{t}|{ts}|{kind}"
                if key in state:
                    continue
                state[key] = now.isoformat()
                pending.append({"t": t, "kind": kind, "dir": d, "detail": detail,
                                "px": float(bars["Close"].iloc[-1]), "bar": ts, "date": day_local})

        # Stage 2: only candidates get the long history needed for EMA5..EMA200
        lines, sent, dropped = [], [], 0
        if pending:
            long_data = fetch(sorted({p["t"] for p in pending}),
                              period="10d" if name == "CRYPTO" else "30d")
            for p in pending:
                need_ema = not (p["kind"] == "EXTREME" and not EXT_REQUIRE_EMA)
                ldf = long_data.get(p["t"])
                closes = completed(ldf, now)["Close"] if ldf is not None else None
                if need_ema and (closes is None or not ema_aligned(closes, p["dir"])):
                    dropped += 1
                    continue
                sym = CRYPTO.get(p["t"], p["t"].replace(".NS", ""))
                icon = "🟢" if p["dir"] == "bull" else "🔴"
                if p["kind"] == "EXTREME":
                    action = "SHORT setup" if p["dir"] == "bear" else "LONG setup"
                    lines.append(f"{icon} <b>{sym}</b> {m['cur']}{p['px']:,.2f}  {kind_label('EXTREME', p['dir'])} - {action} - {p['detail']}")
                else:
                    lines.append(f"{icon} <b>{sym}</b> {m['cur']}{p['px']:,.2f}  {kind_label(p['kind'], p['dir'])} {p['dir'].upper()} - {p['detail']}")
                sent.append({"ts": now.isoformat(), "market": name, "date": p["date"], "bar": p["bar"],
                             "t": p["t"], "sym": sym, "kind": p["kind"], "dir": p["dir"], "px": p["px"]})
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {len(data)} fetched, "
              f"{len(pending)} candidates, {dropped} dropped (EMAs not aligned), "
              f"{len(lines)} alerts, {time.time() - t0:.0f}s")
        if lines:
            for k in range(0, len(lines), 40):
                head = f"<b>🕯️ Live candle alert - {name} (15m, EMA 5/10/20/50/100/200 aligned)</b>\n"
                send(head + "\n".join(lines[k:k + 40]), dry)
            if not dry:
                append_log(sent)
        if not dry:
            maybe_report(name, m, now, state, data, dry)


def sleep_to_next_bar():
    now = datetime.now(timezone.utc)
    nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=15 - now.minute % 15)
    time.sleep(max(1, (nxt + timedelta(seconds=60) - now).total_seconds()))


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
