"""
Crypto Scanner → Telegram
============================
Tracks major cryptocurrencies (BTC, ETH, SOL, BNB, XRP, DOGE) using free
Yahoo Finance data. Crypto trades 24/7, so "today" here just means the
most recent UTC day of bars — same logic as the stock scanners, reused.

One combined message covering: RSI+EMA signal, day high/low proximity,
and trend R² (clean move vs sideways) for each coin.
"""
import json
import os
import time
import urllib.request
import urllib.parse

import yfinance as yf
import pandas as pd
import numpy as np

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
}

SIGNAL_MIN_RSI    = 60
NEAR_EXTREME_PCT  = 0.5
MIN_R2            = 0.70


def ema(series, period):
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(series, period=14):
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def fetch_daily(tickers, period="1y"):
    df = yf.download(list(tickers), period=period, interval="1d",
                      group_by="ticker", threads=True, progress=False, auto_adjust=False)
    out = {}
    for t in tickers:
        try:
            sub = df if len(tickers) == 1 else df[t]
            sub = sub.dropna(how="all")
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue
    return out


def fetch_15m(tickers, period="5d"):
    df = yf.download(list(tickers), period=period, interval="15m",
                      group_by="ticker", threads=True, progress=False, auto_adjust=False)
    out = {}
    for t in tickers:
        try:
            sub = df if len(tickers) == 1 else df[t]
            sub = sub.dropna(how="all")
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue
    return out


def analyze():
    daily = fetch_daily(list(COINS.keys()))
    intraday = fetch_15m(list(COINS.keys()))
    rows = []

    for t, name in COINS.items():
        row = {"sym": name, "ticker": t}

        # Daily signal: RSI + EMA stack
        ddf = daily.get(t)
        if ddf is not None and len(ddf) >= 206:
            closes = ddf["Close"].dropna()
            ltp = float(closes.iloc[-1])
            e9, e20, e50, e100, e200 = (ema(closes, p) for p in (9, 20, 50, 100, 200))
            r = rsi(closes, 14)
            row["ltp"] = round(ltp, 2)
            row["rsi"] = round(r, 1) if r is not None else None
            if None not in (e9, e20, e50, e100, e200):
                row["ema_bullish"] = ltp > e9 > e20 > e50 > e100 > e200
                row["ema_bearish"] = ltp < e9 < e20 < e50 < e100 < e200
            else:
                row["ema_bullish"] = row["ema_bearish"] = None
            row["signal"] = bool(row["ema_bullish"] and r is not None and r >= SIGNAL_MIN_RSI)

        # Intraday: day high/low proximity + trend R^2
        idf = intraday.get(t)
        if idf is not None and not idf.empty:
            idf = idf.dropna(subset=["Close", "High", "Low"])
            today_date = idf.index[-1].date()
            today_bars = idf[idf.index.date == today_date]
            if len(today_bars) >= 5:
                day_high = float(today_bars["High"].max())
                day_low = float(today_bars["Low"].min())
                ltp_i = float(today_bars["Close"].iloc[-1])
                dist_high = (day_high - ltp_i) / day_high * 100 if day_high > 0 else 999
                dist_low = (ltp_i - day_low) / day_low * 100 if day_low > 0 else 999
                row["near_high"] = dist_high <= NEAR_EXTREME_PCT
                row["near_low"] = dist_low <= NEAR_EXTREME_PCT

                closes_i = today_bars["Close"].values.astype(float)
                x = np.arange(len(closes_i))
                slope, intercept = np.polyfit(x, closes_i, 1)
                pred = slope * x + intercept
                ss_res = float(((closes_i - pred) ** 2).sum())
                ss_tot = float(((closes_i - closes_i.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                row["r2"] = round(r2, 2)
                row["trend_dir"] = "up" if slope > 0 else "down"
                row["clean_trend"] = r2 >= MIN_R2

        rows.append(row)

    return rows


def format_message(rows):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["<b>₿ Crypto Scan — BTC/ETH/SOL/BNB/XRP/DOGE</b>", f"{now}", ""]

    for r in rows:
        if "ltp" not in r:
            lines.append(f"<b>{r['sym']}</b>: no data")
            continue
        tags = []
        if r.get("signal"):
            tags.append("✅ SIGNAL (RSI&gt;=60 + EMA bullish)")
        elif r.get("ema_bullish"):
            tags.append("🔼 EMA bullish stack")
        elif r.get("ema_bearish"):
            tags.append("🔽 EMA bearish stack")
        if r.get("near_high"):
            tags.append("at day high")
        if r.get("near_low"):
            tags.append("at day low")
        if r.get("clean_trend"):
            arrow = "🔼" if r.get("trend_dir") == "up" else "🔽"
            tags.append(f"{arrow} clean trend R&#178;={r.get('r2')}")
        tag_str = " · ".join(tags) if tags else "no signal"

        lines.append(f"<b>{r['sym']}</b>  ${r['ltp']:,}  RSI {r.get('rsi')}")
        lines.append(f"  {tag_str}\n")

    return "\n".join(lines)


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main():
    print("Analyzing crypto...")
    rows = analyze()
    for r in rows:
        print(" ", r)
    msg = format_message(rows)
    print("\n--- Telegram message ---")
    print(msg)
    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


STREAK_WINDOW      = 48     # last 48 x 15m candles = 12 hours (crypto has no "session")
STREAK_MIN         = 3
STREAK_MAX_WICK    = 20.0   # closing-side wick <= 20% of the candle's range
STREAK_MIN_AGO     = 1
STREAK_MAX_AGO     = 4
STREAK_MIN_VOL     = 1.0


def streak_for_coin(df):
    """No-wick streak on the last STREAK_WINDOW 15m candles. Returns dict or None.
    Same rules as streak_wick_scan.py (stocks), minus the calendar-day restriction."""
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).iloc[-STREAK_WINDOW:]
    if len(df) < STREAK_MIN + STREAK_MAX_AGO:
        return None
    o, h, l, c = (df[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    v = df["Volume"].values.astype(float)
    n = len(o)

    dirs, ok = [], []
    for i in range(n):
        rng = h[i] - l[i]
        if rng <= 0:
            dirs.append(None); ok.append(False); continue
        red = c[i] < o[i]
        trailing = ((c[i] - l[i]) if red else (h[i] - c[i])) / rng * 100
        dirs.append("red" if red else "green")
        ok.append(trailing <= STREAK_MAX_WICK)

    best_len, best_end, best_dir, run, run_dir = 0, None, None, 0, None
    for i in range(n):
        if ok[i] and dirs[i] == run_dir:
            run += 1
        elif ok[i]:
            run, run_dir = 1, dirs[i]
        else:
            run, run_dir = 0, None
        if run > best_len:
            best_len, best_end, best_dir = run, i, run_dir

    if best_len < STREAK_MIN:
        return None
    ago = (n - 1) - best_end
    if ago < STREAK_MIN_AGO or ago > STREAK_MAX_AGO:
        return None
    day_avg = v.mean()
    vol_ratio = v[best_end - best_len + 1: best_end + 1].mean() / day_avg if day_avg > 0 else 0
    if vol_ratio < STREAK_MIN_VOL:
        return None
    return {"streak": best_len, "ago": ago, "vol": round(vol_ratio, 2),
            "dir": "bullish" if best_dir == "green" else "bearish", "ltp": float(c[-1])}


def run_streak():
    data = fetch_15m(list(COINS.keys()), period="5d")
    lines = ["<b>🕯️ Crypto — No-Wick Streak (15m)</b>",
             f"<i>{STREAK_MIN}+ candles, closing-side wick &lt;={STREAK_MAX_WICK}%, ended {STREAK_MIN_AGO}-{STREAK_MAX_AGO} candles ago, vol&gt;={STREAK_MIN_VOL}x</i>", ""]
    hits = 0
    for t, name in COINS.items():
        df = data.get(t)
        r = streak_for_coin(df) if df is not None else None
        if r:
            hits += 1
            arrow = "🔼 LONG" if r["dir"] == "bullish" else "🔽 SHORT"
            lines.append(f"<b>{name}</b>  ${r['ltp']:,.2f}  {arrow}  {r['streak']}-candle streak ({r['ago']} ago, vol {r['vol']}x)")
    if not hits:
        lines.append("No qualifying streak right now.")
    res = send_telegram("\n".join(lines))
    print(f"streak hits={hits} sent={res.get('ok')}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streak":
        run_streak()
    else:
        main()
