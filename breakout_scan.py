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
MIN_RSI          = 70    # only keep stocks with RSI(14) above this (strong bullish momentum)


def load_universe():
    with open(UNIVERSE_FILE) as f:
        d = json.load(f)
    return d["universe"]


def ema(series, period):
    """Simple EMA over a pandas Series of closes."""
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(series, period=14):
    """Wilder's RSI(14). Returns None if not enough history."""
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


def fetch_daily_bars(tickers, period="1y", interval="1d"):
    """Batch download daily OHLCV for all tickers in chunks (avoids
    Yahoo rate limits on very large ticker lists)."""
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
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

            # RSI(14): only keep stocks with strong bullish momentum
            rsi_14 = rsi(closes, 14)
            if rsi_14 is None or rsi_14 < MIN_RSI:
                continue

            # 20-day momentum: price return over ~1 trading month (shown alongside RSI)
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
                "rsi": round(rsi_14, 1),
                "momentum_pct": momentum_pct,
                "close_above_range": bool(today_close > range_high),
                "ema_stack_ok": True,
            })
        except Exception as e:
            errors += 1
            continue

    # Rank qualifying breakouts by RSI(14) — strongest momentum on top
    breakouts.sort(key=lambda x: -x["rsi"])
    return breakouts, scanned, errors


def format_telegram_message(breakouts, scanned, errors):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🚀 NASDAQ + S&amp;P 500 Breakout Scan</b>", f"{now} · {scanned} scanned",
              f"<i>Filter: 5-day range breakout + EMA 9&gt;20&gt;50&gt;100&gt;200 stack + RSI(14) &gt; {MIN_RSI} · ranked by RSI</i>"]
    if not breakouts:
        lines.append("\nNo breakouts found today.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(breakouts)} breakout(s) found:</b>\n")
    for i, b in enumerate(breakouts[:20], 1):
        hold = "✅ holding" if b["close_above_range"] else "⚠ wick only"
        mom  = f' · 20d {b["momentum_pct"]:+}%' if b["momentum_pct"] is not None else ""
        lines.append(
            f"{i}. <b>{b['sym']}</b>  ${b['close']}  "
            f"(+{b['breakout_pct']}% above 5d range)  {hold}  🔥 RSI {b['rsi']}{mom}"
        )
    return "\n".join(lines)


MAX_LOWER_WICK_PCT = 5.0   # candle low must be within 5% of range from the body's low edge


def scan_no_wick_gap(tickers):
    """Gap-up + no-lower-wick scanner.
    A stock qualifies when today opened ABOVE yesterday's close (gap up,
    any size) AND the candle never traded below its own open (near-zero
    lower wick) — buyers held control the entire session, which tends to
    carry momentum into the next 2-3 days.
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
                continue

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
                continue

            day_pct = (c - o) / o * 100 if o > 0 else 0

            results.append({
                "sym":            t,
                "close":          round(c, 2),
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
    Uses just the FIRST 15-min candle after US market open (9:30-9:45 AM
    ET) so the signal is available right after open, not at end of day.
    """
    data = fetch_daily_bars(tickers, period="5d", interval="15m")
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert("America/New_York")
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


def format_opening_wick_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🌅 NASDAQ + S&amp;P 500 — Opening Gap-Up, No Lower Wick</b>", f"{now} · {scanned} scanned",
             "<i>First 15-min candle after US open gapped up and never dipped below open</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, sorted by gap % (highest first):</b>\n")
    for i, r in enumerate(results[:25], 1):
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ${r['close']}  "
            f"gap +{r['gap_pct']}%  15m {r['candle_pct']:+.2f}%  wick {r['lower_wick_pct']}%"
        )
    return "\n".join(lines)


def format_no_wick_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🎯 NASDAQ + S&amp;P 500 — Gap-Up, No Lower Wick</b>", f"{now} · {scanned} scanned",
             "<i>Gapped up + held above open all session (no dip below open) — tends to carry momentum 2-3 days</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, sorted by gap % (highest first):</b>\n")
    for i, r in enumerate(results[:25], 1):
        body = "🟢" if r["green"] else "🔴"
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ${r['close']}  "
            f"gap +{r['gap_pct']}%  {body} day {r['day_pct']:+.2f}%  wick {r['lower_wick_pct']}%"
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


def scan_smooth_parallel_ema(tickers):
    """Smooth/Parallel EMA5-EMA9 scanner (15-min chart, previous complete day).
    Looks for stocks where the prior trading day's EMA5 and EMA9 moved
    smoothly, in the same direction, without crossing and without the gap
    between them wobbling much — a "quiet coil" that often precedes a
    bigger move the next day.
    """
    data = fetch_daily_bars(tickers, period="10d", interval="15m")
    results, scanned, errors = [], 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert("America/New_York")
            df = df.dropna(subset=["Close"])
            if len(df) < 60:
                continue
            df["ema5"] = df["Close"].ewm(span=5, adjust=False).mean()
            df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()

            all_dates = sorted(set(df.index.date))
            target_date = None
            for d in reversed(all_dates):
                day_bars = df[df.index.date == d]
                if len(day_bars) >= 20:
                    target_date = d
                    break
            if target_date is None:
                continue
            day_bars = df[df.index.date == target_date]
            scanned += 1

            e5, e9 = day_bars["ema5"], day_bars["ema9"]
            spread = e5 - e9
            if not ((spread > 0).all() or (spread < 0).all()):
                continue

            trend5 = float(e5.iloc[-1] - e5.iloc[0])
            trend9 = float(e9.iloc[-1] - e9.iloc[0])
            if trend5 == 0 or (trend5 > 0) != (trend9 > 0):
                continue

            avg_price = float(day_bars["Close"].mean())
            if avg_price <= 0:
                continue
            spread_std_pct = float(spread.std() / avg_price * 100)

            e5_diffs = e5.diff().dropna()
            if len(e5_diffs) < 5 or e5_diffs.abs().mean() == 0:
                continue
            smoothness_cv = float(e5_diffs.std() / e5_diffs.abs().mean())

            day_open  = float(day_bars.iloc[0]["Open"])
            day_close = float(day_bars.iloc[-1]["Close"])
            day_pct = (day_close - day_open) / day_open * 100 if day_open > 0 else 0

            results.append({
                "sym":            t,
                "close":          round(day_close, 2),
                "trend":          "up" if trend5 > 0 else "down",
                "spread_std_pct": round(spread_std_pct, 3),
                "smoothness_cv":  round(smoothness_cv, 2),
                "day_pct":        round(day_pct, 2),
                "date":           str(target_date),
            })
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: x["spread_std_pct"] + x["smoothness_cv"] * 0.1)
    return results, scanned, errors


def format_smooth_parallel_ema_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>〰️ NASDAQ + S&amp;P 500 — Smooth/Parallel EMA5-EMA9 (prev day)</b>", f"{now} · {scanned} scanned",
             "<i>Yesterday's EMA5 &amp; EMA9 moved smoothly, same direction, no crossover — often precedes a bigger move today</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, tightest &amp; smoothest first:</b>\n")
    for i, r in enumerate(results[:25], 1):
        arrow = "🔼" if r["trend"] == "up" else "🔽"
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ${r['close']}  {arrow}  "
            f"tight {r['spread_std_pct']}%  smooth {r['smoothness_cv']}  (yday {r['day_pct']:+.2f}%)"
        )
    return "\n".join(lines)


def main_smooth_ema():
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} stocks for smooth/parallel EMA5-EMA9...")

    results, scanned, errors = scan_smooth_parallel_ema(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

    msg = format_smooth_parallel_ema_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)

    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


SIGNAL_MIN_RSI = 60   # simplified single-signal scan threshold


def scan_signal(tickers):
    """Simplified single-signal scan (daily chart).
    Only two conditions: RSI(14) >= 60, AND price above the full bullish
    EMA stack (9 > 20 > 50 > 100 > 200). Nothing else. Ranked by RSI.
    """
    data = fetch_daily_bars(tickers, period="1y", interval="1d")
    results, scanned, errors = [], 0, 0

    for t, df in data.items():
        try:
            df = df.dropna(subset=["Close"])
            if len(df) < 205:
                continue
            closes = df["Close"]
            ltp = float(closes.iloc[-1])

            scanned += 1

            e9, e20, e50, e100, e200 = (ema(closes, p) for p in (9, 20, 50, 100, 200))
            if None in (e9, e20, e50, e100, e200):
                continue
            if not (ltp > e9 > e20 > e50 > e100 > e200):
                continue

            rsi_14 = rsi(closes, 14)
            if rsi_14 is None or rsi_14 < SIGNAL_MIN_RSI:
                continue

            results.append({"sym": t, "close": round(ltp, 2), "rsi": round(rsi_14, 1)})
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: -x["rsi"])
    return results, scanned, errors


def format_signal_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>✅ NASDAQ + S&amp;P 500 — Signal (RSI&gt;={SIGNAL_MIN_RSI} + EMA stack)</b>",
             f"{now} · {scanned} scanned",
             "<i>Price above EMA 9&gt;20&gt;50&gt;100&gt;200, ranked by RSI</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)
    lines.append(f"\n<b>{len(results)} found:</b>\n")
    for i, r in enumerate(results[:25], 1):
        lines.append(f"{i}. <b>{r['sym']}</b>  ${r['close']}  RSI {r['rsi']}")
    return "\n".join(lines)


def main_signal():
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} stocks for signal (RSI+EMA stack)...")
    results, scanned, errors = scan_signal(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")
    msg = format_signal_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)
    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


MIN_ABOVE_EMA20_PCT = 85    # % of today's 15m bars that must hold above EMA20
MAX_EMA20_DIST_PCT  = 5.0   # max allowed distance from EMA20 (tightness)
MAX_DAY_MOVE_PCT    = 3.0   # max abs day open->close move (flat consolidation)
MIN_BARS_TODAY      = 8     # need a few hours of bars before judging the day


def scan_ema20_hold(tickers):
    """EMA20 hold / tight consolidation scanner (15-min chart).
    Looks for stocks that: broke out above the prior 1-2 day range, then
    spent today consolidating tightly ABOVE a rising 20 EMA (not drifting
    far from it) instead of pulling back hard. This pattern often precedes
    the next leg up.
    """
    data = fetch_daily_bars(tickers, period="10d", interval="15m")
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert("America/New_York")
            df = df.dropna(subset=["Close", "High", "Low", "Open"])
            if len(df) < 40:
                continue

            df["ema20"] = df["Close"].ewm(span=20, adjust=False).mean()

            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            prior_bars = df[df.index.date < today_date]
            if len(today_bars) < MIN_BARS_TODAY or prior_bars.empty:
                continue

            scanned += 1

            above_frac = (today_bars["Low"] >= today_bars["ema20"]).mean() * 100
            if above_frac < MIN_ABOVE_EMA20_PCT:
                continue

            dist_pct = (today_bars["Close"] - today_bars["ema20"]) / today_bars["ema20"] * 100
            max_dist = float(dist_pct.max())
            avg_dist = float(dist_pct.mean())
            if max_dist > MAX_EMA20_DIST_PCT:
                continue

            prior_dates = sorted(set(prior_bars.index.date))[-2:]
            prior_mask  = pd.Series(prior_bars.index.date, index=prior_bars.index).isin(prior_dates)
            prior_high  = prior_bars.loc[prior_mask, "High"].max()
            today_low_all = float(today_bars["Low"].min())
            if pd.isna(prior_high) or today_low_all < float(prior_high):
                continue

            day_open  = float(today_bars.iloc[0]["Open"])
            day_close = float(today_bars.iloc[-1]["Close"])
            if day_open <= 0:
                continue
            day_pct = (day_close - day_open) / day_open * 100
            if abs(day_pct) > MAX_DAY_MOVE_PCT:
                continue

            results.append({
                "sym":          t,
                "close":        round(day_close, 2),
                "ema20":        round(float(today_bars["ema20"].iloc[-1]), 2),
                "avg_dist_pct": round(avg_dist, 2),
                "max_dist_pct": round(max_dist, 2),
                "day_pct":      round(day_pct, 2),
                "above_frac":   round(above_frac, 1),
                "prior_high":   round(float(prior_high), 2),
            })
        except Exception:
            errors += 1
            continue

    results.sort(key=lambda x: x["avg_dist_pct"])
    return results, scanned, errors


def format_ema20_hold_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>📐 NASDAQ + S&amp;P 500 — EMA20 Hold / Tight Consolidation</b>", f"{now} · {scanned} scanned",
             "<i>Above prior 1-2 day range, holding tight above rising 20 EMA (15m), flat day</i>"]
    if not results:
        lines.append("\nNo qualifying stocks found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} found, tightest first:</b>\n")
    for i, r in enumerate(results[:25], 1):
        lines.append(
            f"{i}. <b>{r['sym']}</b>  ${r['close']}  "
            f"EMA20 ${r['ema20']}  dist {r['avg_dist_pct']}%  day {r['day_pct']:+.2f}%"
        )
    return "\n".join(lines)


def main_ema20_hold():
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} stocks for EMA20 hold consolidation...")

    results, scanned, errors = scan_ema20_hold(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

    msg = format_ema20_hold_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)

    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


def main_no_wick():
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} stocks for gap-up/no-lower-wick...")

    results, scanned, errors = scan_no_wick_gap(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

    msg = format_no_wick_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)

    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


def main_open_wick():
    print("Loading universe...")
    tickers = load_universe()
    print(f"Scanning {len(tickers)} stocks opening candle gap-up/no-wick...")

    results, scanned, errors = scan_opening_wick(tickers)
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")

    msg = format_opening_wick_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)

    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "no_wick":
        main_no_wick()
    elif len(sys.argv) > 1 and sys.argv[1] == "open_wick":
        main_open_wick()
    elif len(sys.argv) > 1 and sys.argv[1] == "ema20_hold":
        main_ema20_hold()
    elif len(sys.argv) > 1 and sys.argv[1] == "smooth_ema":
        main_smooth_ema()
    elif len(sys.argv) > 1 and sys.argv[1] == "signal":
        main_signal()
    else:
        main()
