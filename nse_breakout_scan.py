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


def fetch_daily_bars(tickers, period="1y", interval="1d"):
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = [f"{t}.NS" for t in tickers[i:i + chunk_size]]
        try:
            df = yf.download(
                chunk, period=period, interval=interval,
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


MAX_LOWER_WICK_PCT = 5.0   # candle low must be within 5% of range from the body's low edge


def scan_no_wick_gap(tickers):
    """Gap-up + no-lower-wick scanner.
    A stock qualifies when today opened ABOVE yesterday's close (gap up,
    any size) AND the candle never traded below its own open (near-zero
    lower wick) — a sign buyers held control the entire session, which
    tends to carry momentum into the next 2-3 days.
    Sorted by gap % descending (biggest gap-up-and-held first).
    """
    data = fetch_daily_bars(tickers, period="5d")
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if len(df) < 2:
                continue
            today      = df.iloc[-1]
            prev_close = float(df.iloc[-2]["Close"])
            o, h, l, c = (float(today["Open"]), float(today["High"]),
                          float(today["Low"]), float(today["Close"]))
            if any(pd.isna(x) for x in (o, h, l, c, prev_close)):
                continue
            if prev_close <= 0 or o <= 0:
                continue

            scanned += 1

            gap_pct = (o - prev_close) / prev_close * 100
            if gap_pct <= 0:
                continue   # not a gap up

            rng = h - l
            if rng <= 0:
                continue
            # Sanity-check the candle: reject corrupted OHLC data (e.g. a
            # bad row where open/low don't make sense relative to high/low)
            if not (l <= o <= h and l <= c <= h):
                continue
            # Measured against the OPEN, not the candle body — a stock that
            # gapped up then faded to close red should NOT qualify, since
            # that's giving back the gap, not holding it.
            lower_wick_pct = (o - l) / rng * 100
            if lower_wick_pct < 0 or lower_wick_pct > MAX_LOWER_WICK_PCT or c < o:
                continue   # dipped below open, or closed red — doesn't qualify

            day_pct = (c - o) / o * 100 if o > 0 else 0

            results.append({
                "sym":            t,
                "close":          round(c, 2),
                "open":           round(o, 2),
                "gap_pct":        round(gap_pct, 2),
                "day_pct":        round(day_pct, 2),
                "lower_wick_pct": round(lower_wick_pct, 1),
                "green":          bool(c > o),
            })
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: -x["gap_pct"])
    return results, scanned, errors


def scan_opening_wick(tickers):
    """Opening-candle gap-up + no-lower-wick scanner.
    Uses just the FIRST 15-min candle of today (9:15-9:30 AM IST) so the
    signal is available right after market open, not at end of day.
    Same idea as scan_no_wick_gap, but checkable within minutes of the open.
    """
    data = fetch_daily_bars(tickers, period="5d", interval="15m")
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert("Asia/Kolkata")
            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            prior_bars = df[df.index.date < today_date]
            if today_bars.empty or prior_bars.empty:
                continue

            first_bar  = today_bars.iloc[0]
            prev_close = float(prior_bars.iloc[-1]["Close"])
            o, h, l, c = (float(first_bar["Open"]), float(first_bar["High"]),
                          float(first_bar["Low"]), float(first_bar["Close"]))
            if any(pd.isna(x) for x in (o, h, l, c, prev_close)):
                continue
            if prev_close <= 0 or o <= 0:
                continue

            scanned += 1

            gap_pct = (o - prev_close) / prev_close * 100
            if gap_pct <= 0:
                continue

            rng = h - l
            if rng <= 0:
                continue
            if not (l <= o <= h and l <= c <= h):
                continue
            lower_wick_pct = (o - l) / rng * 100
            if lower_wick_pct < 0 or lower_wick_pct > MAX_LOWER_WICK_PCT or c < o:
                continue

            candle_pct = (c - o) / o * 100 if o > 0 else 0

            results.append({
                "sym":            t,
                "close":          round(c, 2),
                "gap_pct":        round(gap_pct, 2),
                "candle_pct":     round(candle_pct, 2),
                "lower_wick_pct": round(lower_wick_pct, 1),
            })
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: -x["gap_pct"])
    return results, scanned, errors


def format_opening_wick_message(index_key, results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = INDEX_NAMES[index_key]
    lines = [f"<b>🌅 {name} — Opening Gap-Up, No Lower Wick</b>", f"{now} · {scanned} scanned",
             "<i>First 15-min candle (9:15-9:30 AM) gapped up and never dipped below open</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, sorted by gap % (highest first):</b>\n")
    for i, r in enumerate(results[:20], 1):
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ₹{r['close']}  "
            f"gap +{r['gap_pct']}%  15m {r['candle_pct']:+.2f}%  wick {r['lower_wick_pct']}%"
        )
    return "\n".join(lines)


def format_no_wick_message(index_key, results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    name = INDEX_NAMES[index_key]
    lines = [f"<b>🎯 {name} — Gap-Up, No Lower Wick</b>", f"{now} · {scanned} scanned",
             "<i>Gapped up + held above open all session (no dip below open) — tends to carry momentum 2-3 days</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, sorted by gap % (highest first):</b>\n")
    for i, r in enumerate(results[:20], 1):
        body = "🟢" if r["green"] else "🔴"
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ₹{r['close']}  "
            f"gap +{r['gap_pct']}%  {body} day {r['day_pct']:+.2f}%  wick {r['lower_wick_pct']}%"
        )
    return "\n".join(lines)


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


def main_no_wick():
    indices = load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        print(f"\nScanning {INDEX_NAMES[key]} ({len(tickers)} stocks) for gap-up/no-lower-wick...")
        results, scanned, errors = scan_no_wick_gap(tickers)
        print(f"  Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

        msg = format_no_wick_message(key, results, scanned)
        result = send_telegram(msg)
        print(f"  Telegram send result: {result.get('ok')}")
        time.sleep(1)


def main_open_wick():
    indices = load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        print(f"\nScanning {INDEX_NAMES[key]} ({len(tickers)} stocks) opening candle gap-up/no-wick...")
        results, scanned, errors = scan_opening_wick(tickers)
        print(f"  Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

        msg = format_opening_wick_message(key, results, scanned)
        result = send_telegram(msg)
        print(f"  Telegram send result: {result.get('ok')}")
        time.sleep(1)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "no_wick":
        main_no_wick()
    elif len(sys.argv) > 1 and sys.argv[1] == "open_wick":
        main_open_wick()
    else:
        main()
