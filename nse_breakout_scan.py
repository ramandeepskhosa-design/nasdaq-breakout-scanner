"""
NSE 5-Day Range Breakout Scanner → Telegram (Nifty 50 / Next 50 / Midcap 50 / Smallcap 250)
=============================================================================================
Same logic as breakout_scan.py but for Indian stocks via free Yahoo Finance
data (ticker + ".NS"), so no Zerodha login/token is needed at all.

Smallcap 250 has too many names, so it's additionally filtered to only
stocks trading above the full EMA 9/20/50/100/200 bullish stack — same
as the local Zerodha-based scanner at localhost:8080.
"""
import json
import os
import time
import urllib.request
import urllib.parse

import yfinance as yf
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE  = os.path.join(BASE_DIR, "nse_index_stocks.json")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

MIN_BREAKOUT_PCT = 0.3

INDEX_NAMES = {
    "nifty50":      "Nifty 50",
    "nifty_next50": "Nifty Next 50",
    "midcap50":     "Midcap 50",
    "smallcap250":  "Smallcap 250",
}


def load_indices():
    with open(INDEX_FILE) as f:
        return json.load(f)


def ema(series, period):
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def fetch_daily_bars(tickers, period="1y"):
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = [f"{t}.NS" for t in tickers[i:i + chunk_size]]
        try:
            df = yf.download(
                chunk, period=period, interval="1d",
                group_by="ticker", threads=True, progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"  chunk {i}-{i+chunk_size} failed: {e}")
            continue
        for t in chunk:
            try:
                sub = df if len(chunk) == 1 else df[t]
                sub = sub.dropna(how="all")
                if not sub.empty:
                    all_data[t.replace(".NS", "")] = sub
            except Exception:
                continue
        time.sleep(1)
    return all_data


def scan_breakouts(tickers, apply_ema_filter=False):
    data = fetch_daily_bars(tickers)
    breakouts = []
    scanned, errors = 0, 0
    min_days = 205 if apply_ema_filter else 6

    for t, df in data.items():
        try:
            if len(df) < min_days:
                continue
            today   = df.iloc[-1]
            history = df.iloc[:-1].tail(5)
            if len(history) < 5:
                continue

            range_high  = history["High"].max()
            today_high  = today["High"]
            today_open  = today["Open"]
            today_close = today["Close"]

            scanned += 1

            if range_high <= 0 or today_high <= range_high:
                continue

            breakout_pct = round((today_high - range_high) / range_high * 100, 2)
            if breakout_pct < MIN_BREAKOUT_PCT:
                continue

            if apply_ema_filter:
                closes = df["Close"]
                e9, e20, e50, e100, e200 = (ema(closes, p) for p in (9, 20, 50, 100, 200))
                if None in (e9, e20, e50, e100, e200):
                    continue
                if not (today_close > e9 > e20 > e50 > e100 > e200):
                    continue

            day_pct = round((today_close - today_open) / today_open * 100, 2) if today_open > 0 else 0

            breakouts.append({
                "sym":              t,
                "close":            round(float(today_close), 2),
                "today_high":       round(float(today_high), 2),
                "range_high":       round(float(range_high), 2),
                "breakout_pct":     breakout_pct,
                "day_pct":          day_pct,
                "close_above_range": bool(today_close > range_high),
            })
        except Exception:
            errors += 1
            continue

    breakouts.sort(key=lambda x: -x["breakout_pct"])
    return breakouts, scanned, errors


def format_message(index_key, breakouts, scanned, ema_filtered):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = INDEX_NAMES[index_key]
    lines = [f"<b>🚀 {name} — 5-Day Breakout Scan</b>", f"{now} · {scanned} scanned"]
    if ema_filtered:
        lines.append("<i>Filter: close above EMA 9&gt;20&gt;50&gt;100&gt;200 (full bullish stack)</i>")
    if not breakouts:
        lines.append("\nNo breakouts found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(breakouts)} breakout(s) found:</b>\n")
    for i, b in enumerate(breakouts[:20], 1):
        hold = "✅ holding" if b["close_above_range"] else "⚠ wick only"
        lines.append(
            f"{i}. <b>{b['sym']}</b>  ₹{b['close']}  "
            f"(+{b['breakout_pct']}% above 5d range)  {hold}"
        )
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
    indices = load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        apply_ema = (key == "smallcap250")
        print(f"\nScanning {INDEX_NAMES[key]} ({len(tickers)} stocks, EMA filter={apply_ema})...")
        breakouts, scanned, errors = scan_breakouts(tickers, apply_ema_filter=apply_ema)
        print(f"  Scanned: {scanned}  Errors: {errors}  Breakouts: {len(breakouts)}")

        msg = format_message(key, breakouts, scanned, apply_ema)
        result = send_telegram(msg)
        print(f"  Telegram send result: {result.get('ok')}")
        time.sleep(1)   # avoid Telegram rate limit between messages


if __name__ == "__main__":
    main()
