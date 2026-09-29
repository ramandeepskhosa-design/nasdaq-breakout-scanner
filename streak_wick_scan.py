"""
Institutional Streak Scanner (Small Trailing Wick) → Telegram
==================================================================
Finds stocks with 3+ consecutive same-direction 15-min candles where the
closing-side wick stays small (lower wick for red candles, upper wick for
green) — price keeps closing at/near its extreme each candle, no real
pushback. That's the signature of sustained institutional participation
vs choppy, indecisive candles.

Best run near end of day (~3 PM IST) to judge which stocks are showing
real conviction worth carrying into tomorrow's trade.
"""
import os
import time

import trend_extreme_scan as tex
import pandas as pd

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

MAX_TRAILING_WICK = 20.0   # closing-side wick must stay <= this % of the candle's range
MIN_STREAK        = 3      # need at least this many consecutive qualifying candles
MAX_CANDLES_AGO   = 6      # streak must have ended within the last N candles (stay relevant)


def scan_streak(tickers, tz):
    data = tex.fetch_15m(tickers)
    results = []
    scanned = 0

    for t, df in data.items():
        try:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            if df.index.tz is not None:
                df = df.tz_convert(tz)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if df.empty:
                continue
            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            if len(today_bars) < MIN_STREAK:
                continue
            scanned += 1

            o = today_bars["Open"].values.astype(float)
            h = today_bars["High"].values.astype(float)
            l = today_bars["Low"].values.astype(float)
            c = today_bars["Close"].values.astype(float)
            n = len(o)

            dirs, qualifies = [], []
            for i in range(n):
                rng = h[i] - l[i]
                if rng <= 0:
                    dirs.append(None); qualifies.append(False); continue
                is_red = c[i] < o[i]
                trailing_wick = ((c[i] - l[i]) if is_red else (h[i] - c[i])) / rng * 100
                dirs.append("red" if is_red else "green")
                qualifies.append(trailing_wick <= MAX_TRAILING_WICK)

            best_len, best_end, best_dir = 0, None, None
            run_len, run_dir = 0, None
            for i in range(n):
                if qualifies[i] and dirs[i] == run_dir:
                    run_len += 1
                elif qualifies[i]:
                    run_dir = dirs[i]
                    run_len = 1
                else:
                    run_len, run_dir = 0, None
                if run_len > best_len:
                    best_len, best_end, best_dir = run_len, i, run_dir

            if best_len < MIN_STREAK:
                continue
            candles_ago = (n - 1) - best_end
            if candles_ago > MAX_CANDLES_AGO:
                continue

            ltp = float(c[-1])
            results.append({
                "sym": t.replace(".NS", ""), "close": round(ltp, 2),
                "direction": "bullish" if best_dir == "green" else "bearish",
                "streak": best_len, "candles_ago": candles_ago,
            })
        except Exception:
            continue

    results.sort(key=lambda x: (-x["streak"], x["candles_ago"]))
    return results, scanned


def format_message(title, results, scanned, currency="₹"):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🕯️ {title} — Institutional Streak</b>", f"{now} · {scanned} scanned",
             f"<i>{MIN_STREAK}+ candles, closing-side wick &lt;={MAX_TRAILING_WICK}%, within last {MAX_CANDLES_AGO} candles</i>", ""]
    if not results:
        lines.append("None found.")
        return "\n".join(lines)
    lines.append(f"<b>{len(results)} found:</b>\n")
    for i, r in enumerate(results[:30], 1):
        arrow = "🔼 CE" if r["direction"] == "bullish" else "🔽 PE"
        lines.append(f"{i}. <b>{r['sym']}</b>  {currency}{r['close']}  {arrow}  {r['streak']}-candle streak ({r['candles_ago']} ago)")
    return "\n".join(lines)


def run_nse():
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    universe = list(dict.fromkeys(indices["nifty50"] + indices["nifty_next50"] + indices["midcap50"]))
    tickers = [f"{s}.NS" for s in universe]
    results, scanned = scan_streak(tickers, "Asia/Kolkata")
    print(f"NSE (Nifty50+Next50+Midcap50): scanned={scanned} found={len(results)}")
    msg = format_message("Nifty 50 + Next 50 + Midcap 50", results, scanned, "₹")
    r = tex.send_telegram(msg)
    print("sent:", r.get("ok"))


def run_nasdaq():
    from breakout_scan import load_universe
    tickers = load_universe()
    print(f"Scanning {len(tickers)} NASDAQ+S&P stocks...")
    results, scanned = scan_streak(tickers, "America/New_York")
    print(f"scanned={scanned} found={len(results)}")
    msg = format_message("NASDAQ + S&P 500", results, scanned, "$")
    r = tex.send_telegram(msg)
    print("sent:", r.get("ok"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "nasdaq":
        run_nasdaq()
    else:
        run_nse()
