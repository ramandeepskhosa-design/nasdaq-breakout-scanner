"""
Trending-at-Extreme Scanner → Telegram
=========================================
Finds stocks trading right at today's high or low, where the day's move
was a clean, consistent trend (not sideways chop). Uses linear regression
R^2 on the day's 15-min closes: R^2 close to 1 means price moved in an
almost straight line; low R^2 means it whipsawed/consolidated.

Works for both NSE (indices) and NASDAQ + S&P 500.
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

NEAR_EXTREME_PCT = 0.5   # must be within this % of day's high/low
MIN_R2           = 0.70  # min trend "straightness" (0-1, higher = smoother)
MIN_BARS_TODAY   = 10


def fetch_15m(tickers, period="5d"):
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = yf.download(
                chunk, period=period, interval="15m",
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
                    all_data[t] = sub
            except Exception:
                continue
        time.sleep(1)
    return all_data


def scan_trending_at_extreme(tickers, tz="Asia/Kolkata", currency="₹"):
    data = fetch_15m(tickers)
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert(tz)
            df = df.dropna(subset=["Close", "High", "Low"])
            if df.empty:
                continue

            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            if len(today_bars) < MIN_BARS_TODAY:
                continue

            scanned += 1

            closes  = today_bars["Close"]
            day_high = float(today_bars["High"].max())
            day_low  = float(today_bars["Low"].min())
            ltp      = float(closes.iloc[-1])
            if day_high <= 0 or day_low <= 0:
                continue

            dist_from_high = (day_high - ltp) / day_high * 100
            dist_from_low  = (ltp - day_low) / day_low * 100

            near_high = dist_from_high <= NEAR_EXTREME_PCT
            near_low  = dist_from_low <= NEAR_EXTREME_PCT
            if not (near_high or near_low):
                continue

            # Linear regression R^2 of closes vs bar index — trend "straightness"
            x = np.arange(len(closes))
            y = closes.values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            if r2 < MIN_R2:
                continue

            # Direction must match which extreme it's near (near high + rising, or near low + falling)
            if near_high and slope > 0:
                direction, at = "rising", "high"
            elif near_low and slope < 0:
                direction, at = "falling", "low"
            else:
                continue

            day_open = float(today_bars.iloc[0]["Open"])
            day_pct = (ltp - day_open) / day_open * 100 if day_open > 0 else 0

            results.append({
                "sym": t, "close": round(ltp, 2), "direction": direction, "at": at,
                "r2": round(r2, 2), "day_pct": round(day_pct, 2),
            })
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: -x["r2"])
    return results, scanned, errors


def format_message(title, results, scanned, currency="₹"):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>📉📈 {title} — Trending at Day Extreme</b>", f"{now} · {scanned} scanned",
             f"<i>Within {NEAR_EXTREME_PCT}% of day high/low, R&#178;&gt;={MIN_R2} (clean trend, not sideways)</i>"]
    if not results:
        lines.append("\nNone found.")
        return "\n".join(lines)
    lines.append(f"\n<b>{len(results)} found, smoothest trend first:</b>\n")
    for i, r in enumerate(results[:25], 1):
        arrow = "🔼" if r["direction"] == "rising" else "🔽"
        lines.append(
            f"{i}. <b>{r['sym']}</b>  {currency}{r['close']}  {arrow} at day {r['at']}  "
            f"R&#178;={r['r2']}  day {r['day_pct']:+.2f}%"
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


def main_nse():
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        print(f"\nScanning {name} ({len(tickers)} stocks)...")
        results, scanned, errors = scan_trending_at_extreme(tickers, tz="Asia/Kolkata")
        for r in results:
            r["sym"] = r["sym"].replace(".NS", "")
        print(f"  Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")
        msg = format_message(name, results, scanned, currency="₹")
        result = send_telegram(msg)
        print(f"  Telegram send result: {result.get('ok')}")
        time.sleep(1)


def main_nasdaq():
    from breakout_scan import load_universe
    tickers = load_universe()
    print(f"\nScanning NASDAQ + S&P 500 ({len(tickers)} stocks)...")
    results, scanned, errors = scan_trending_at_extreme(tickers, tz="America/New_York")
    print(f"  Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")
    msg = format_message("NASDAQ + S&P 500", results, scanned, currency="$")
    result = send_telegram(msg)
    print(f"  Telegram send result: {result.get('ok')}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "nasdaq":
        main_nasdaq()
    else:
        main_nse()
