"""
NASDAQ + S&P 500 Range Breakout Scanner → Telegram
====================================================
Scans a merged NASDAQ + S&P 500 universe (~700 stocks) using free Yahoo
Finance data. A stock "breaks out" when its most recent daily high trades
above the highest daily high of the previous 5 trading days, AND it's
trading above the full EMA 9>20>50>100>200 bullish stack.

Qualifying breakouts are ranked by 20-day momentum (price return over the
last ~1 month) so the strongest movers show first.

Sends the top breakouts straight to a Telegram bot — no server, no
laptop needed once scheduled.

Run manually:  python3 breakout_scan.py
"""
import json
import os
import sys
import time
import urllib.request
import urllib.parse

import yfinance as yf
import pandas as pd

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UNIVERSE_FILE = os.path.join(BASE_DIR, "universe.json")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

MIN_BREAKOUT_PCT = 0.3   # ignore noise below this


def load_universe():
    with open(UNIVERSE_FILE) as f:
        d = json.load(f)
    return d["universe"]


def ema(series, period):
    """Simple EMA over a pandas Series of closes."""
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def fetch_daily_bars(tickers, period="1y"):
    """Batch download daily OHLCV for all tickers in chunks (avoids
    Yahoo rate limits on very large ticker lists)."""
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
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
                if len(chunk) == 1:
                    sub = df
                else:
                    sub = df[t]
                sub = sub.dropna(how="all")
                if not sub.empty:
                    all_data[t] = sub
            except Exception:
                continue
        time.sleep(1)  # be polite to Yahoo
    return all_data


def scan_breakouts(tickers):
    data = fetch_daily_bars(tickers)
    breakouts = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if len(df) < 205:
                continue   # need ~200 trading days for EMA 200
            today   = df.iloc[-1]
            history = df.iloc[:-1].tail(5)   # previous 5 trading days
            if len(history) < 5:
                continue

            range_high = history["High"].max()
            today_high = today["High"]
            today_open = today["Open"]
            today_close = today["Close"]

            scanned += 1

            if range_high <= 0 or today_high <= range_high:
                continue

            breakout_pct = round((today_high - range_high) / range_high * 100, 2)
            if breakout_pct < MIN_BREAKOUT_PCT:
                continue

            closes = df["Close"]
            e9, e20, e50, e100, e200 = (ema(closes, p) for p in (9, 20, 50, 100, 200))
            if None in (e9, e20, e50, e100, e200):
                continue
            ema_stack_ok = today_close > e9 > e20 > e50 > e100 > e200
            if not ema_stack_ok:
                continue

            day_pct = round((today_close - today_open) / today_open * 100, 2) if today_open > 0 else 0

            # 20-day momentum: price return over ~1 trading month
            momentum_pct = None
            if len(closes) >= 21:
                close_20d_ago = float(closes.iloc[-21])
                if close_20d_ago > 0:
                    momentum_pct = round((float(today_close) - close_20d_ago) / close_20d_ago * 100, 2)

            breakouts.append({
                "sym": t,
                "close": round(float(today_close), 2),
                "today_high": round(float(today_high), 2),
                "range_high": round(float(range_high), 2),
                "breakout_pct": breakout_pct,
                "day_pct": day_pct,
                "momentum_pct": momentum_pct,
                "close_above_range": bool(today_close > range_high),
                "ema_stack_ok": True,
            })
        except Exception as e:
            errors += 1
            continue

    # Rank qualifying breakouts by 20-day momentum first (strongest movers on top)
    breakouts.sort(key=lambda x: -(x["momentum_pct"] if x["momentum_pct"] is not None else -999))
    return breakouts, scanned, errors


def format_telegram_message(breakouts, scanned, errors):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🚀 NASDAQ + S&amp;P 500 Breakout Scan</b>", f"{now} · {scanned} scanned",
              "<i>Filter: 5-day range breakout + EMA 9&gt;20&gt;50&gt;100&gt;200 stack · ranked by 20-day momentum</i>"]
    if not breakouts:
        lines.append("\nNo breakouts found today.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(breakouts)} breakout(s) found:</b>\n")
    for i, b in enumerate(breakouts[:20], 1):
        hold = "✅ holding" if b["close_above_range"] else "⚠ wick only"
        mom  = f' · 🔥 +{b["momentum_pct"]}% (20d)' if b["momentum_pct"] is not None else ""
        lines.append(
            f"{i}. <b>{b['sym']}</b>  ${b['close']}  "
            f"(+{b['breakout_pct']}% above 5d range)  {hold}{mom}"
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
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} NASDAQ stocks for 5-day range breakouts...")

    breakouts, scanned, errors = scan_breakouts(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Breakouts: {len(breakouts)}")

    msg = format_telegram_message(breakouts, scanned, errors)
    print("\n--- Telegram message ---")
    print(msg)

    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))
    if not result.get("ok"):
        print(result)


if __name__ == "__main__":
    main()
