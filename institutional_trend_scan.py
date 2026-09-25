"""
Institutional Trend (Full-Body / Low-Wick) Scanner → Telegram
=================================================================
Real institutional buying/selling tends to show as several consecutive
FULL-BODY candles (small wicks) making higher-highs+higher-lows (buying)
or lower-lows+lower-highs (selling) — not choppy, wicky candles, which
signal indecision/low conviction.

Looks at the last WINDOW 15-min candles of the day (run this near EOD,
e.g. 2-3 PM IST, to judge which stocks are likely to continue their move
into tomorrow). Works for NSE indices and NASDAQ + S&P 500.
"""
import os
import time

import trend_extreme_scan as tex

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

WINDOW           = 8     # last N 15-min candles to judge (~2 hours)
MIN_BODY_RATIO   = 55.0  # avg body must be >=55% of each candle's range
MIN_STRUCTURE    = 0.55  # fraction of candles making HH+HL / LL+LH
MIN_DIRECTION    = 0.6   # fraction of candles that must be up (or down)


def scan_institutional_trend(tickers, tz):
    data = tex.fetch_15m(tickers)
    results = []
    scanned, errors = 0, 0

    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert(tz)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if df.empty:
                continue
            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            if len(today_bars) < WINDOW:
                continue

            scanned += 1
            window = today_bars.iloc[-WINDOW:]
            o = window["Open"].values.astype(float)
            h = window["High"].values.astype(float)
            l = window["Low"].values.astype(float)
            c = window["Close"].values.astype(float)
            k = len(o)

            body_ratios = []
            for i in range(k):
                rng = h[i] - l[i]
                body_ratios.append(abs(c[i] - o[i]) / rng * 100 if rng > 0 else 0)
            avg_body = sum(body_ratios) / k

            up = sum(1 for i in range(k) if c[i] > o[i])
            down = sum(1 for i in range(k) if c[i] < o[i])

            hh_hl = sum(1 for i in range(1, k) if h[i] > h[i-1] and l[i] > l[i-1])
            ll_lh = sum(1 for i in range(1, k) if l[i] < l[i-1] and h[i] < h[i-1])
            structure_frac_bull = hh_hl / (k - 1)
            structure_frac_bear = ll_lh / (k - 1)

            is_bull = (up / k >= MIN_DIRECTION) and (structure_frac_bull >= MIN_STRUCTURE) and (avg_body >= MIN_BODY_RATIO)
            is_bear = (down / k >= MIN_DIRECTION) and (structure_frac_bear >= MIN_STRUCTURE) and (avg_body >= MIN_BODY_RATIO)

            if not (is_bull or is_bear):
                continue

            direction = "bullish" if is_bull else "bearish"
            structure_frac = structure_frac_bull if is_bull else structure_frac_bear
            ltp = float(c[-1])
            day_open = float(today_bars.iloc[0]["Open"])
            day_pct = (ltp - day_open) / day_open * 100 if day_open > 0 else 0

            results.append({
                "sym": t.replace(".NS", ""),
                "close": round(ltp, 2),
                "direction": direction,
                "avg_body": round(avg_body, 1),
                "structure": round(structure_frac, 2),
                "day_pct": round(day_pct, 2),
            })
        except Exception:
            errors += 1
            continue

    # Cleanest conviction first: high structure + high body ratio
    results.sort(key=lambda x: -(x["structure"] * x["avg_body"]))
    return results, scanned, errors


def format_message(title, results, scanned, currency="₹"):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🏦 {title} — Institutional Trend (Full-Body, Low-Wick)</b>", f"{now} · {scanned} scanned",
             f"<i>Last {WINDOW} candles: body&gt;={MIN_BODY_RATIO}%, structure&gt;={MIN_STRUCTURE}, direction&gt;={MIN_DIRECTION}</i>"]
    if not results:
        lines.append("\nNone found.")
        return "\n".join(lines)
    lines.append(f"\n<b>{len(results)} found, cleanest conviction first:</b>\n")
    for i, r in enumerate(results[:25], 1):
        arrow = "🔼" if r["direction"] == "bullish" else "🔽"
        lines.append(
            f"{i}. <b>{r['sym']}</b>  {currency}{r['close']}  {arrow}  "
            f"body {r['avg_body']}%  structure {r['structure']}  day {r['day_pct']:+.2f}%"
        )
    return "\n".join(lines)


def run_nse():
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        results, scanned, errors = scan_institutional_trend(tickers, "Asia/Kolkata")
        print(f"{name}: scanned={scanned} errors={errors} found={len(results)}")
        msg = format_message(name, results, scanned, "₹")
        r = tex.send_telegram(msg)
        print("  sent:", r.get("ok"))
        time.sleep(1)


def run_nasdaq():
    from breakout_scan import load_universe
    tickers = load_universe()
    print(f"Scanning {len(tickers)} NASDAQ+S&P stocks...")
    results, scanned, errors = scan_institutional_trend(tickers, "America/New_York")
    print(f"scanned={scanned} errors={errors} found={len(results)}")
    msg = format_message("NASDAQ + S&P 500", results, scanned, "$")
    r = tex.send_telegram(msg)
    print("sent:", r.get("ok"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "nasdaq":
        run_nasdaq()
    else:
        run_nse()
