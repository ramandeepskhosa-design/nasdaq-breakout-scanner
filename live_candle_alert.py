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

IMPULSE_MIN_BODY = 80.0
IMPULSE_MULT     = 3.0
AVG_LEN          = 20
NW_MAX_WICK      = 20.0
NW_STREAK        = 3
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
                   "hours": ((9, 15), (15, 35)), "last_bar": (15, 15), "cur": "₹"},
        "US":     {"tickers": uni, "tz": "America/New_York",
                   "hours": ((9, 30), (16, 5)), "last_bar": (15, 45), "cur": "$"},
        "CRYPTO": {"tickers": list(CRYPTO), "tz": "UTC", "hours": None, "cur": "$"},
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
            lb = m.get("last_bar")
            if lb:
                st = bars.index[-1].astimezone(ZoneInfo(m["tz"]))
                if (st.hour, st.minute) == lb:
                    continue   # closing-drift candle: full body on thin volume, mostly noise
            for kind, d, detail in evaluate(bars):
                key = f"{t}|{ts}|{kind}"
                if key in state:
                    continue
                state[key] = now.isoformat()
                pending.append((t, kind, d, detail, float(bars["Close"].iloc[-1])))

        # Stage 2: only candidates get the long history needed for EMA5..EMA200
        lines, dropped = [], 0
        if pending:
            long_data = fetch(sorted({p[0] for p in pending}),
                              period="10d" if name == "CRYPTO" else "30d")
            for t, kind, d, detail, px in pending:
                ldf = long_data.get(t)
                closes = completed(ldf, now)["Close"] if ldf is not None else None
                if closes is None or not ema_aligned(closes, d):
                    dropped += 1
                    continue
                sym = CRYPTO.get(t, t.replace(".NS", ""))
                icon = "🟢" if d == "bull" else "🔴"
                label = "IMPULSE" if kind == "IMPULSE" else "NO-WICK"
                lines.append(f"{icon} <b>{sym}</b> {m['cur']}{px:,.2f}  {label} {d.upper()} - {detail}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {name}: {len(data)} fetched, "
              f"{len(pending)} candidates, {dropped} dropped (EMAs not aligned), "
              f"{len(lines)} alerts, {time.time() - t0:.0f}s")
        if lines:
            for k in range(0, len(lines), 40):
                head = f"<b>🕯️ Live candle alert - {name} (15m, EMA 5/10/20/50/100/200 aligned)</b>\n"
                send(head + "\n".join(lines[k:k + 40]), dry)


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
