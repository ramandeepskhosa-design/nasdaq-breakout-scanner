"""
At Day High / Day Low Scanner → Telegram
===========================================
Plain list of stocks currently within NEAR_PCT of today's high or low —
no trend/smoothness filter (see trend_extreme_scan.py for the R^2-based
"clean trend, not sideways" version). Works for NSE (indices) and
NASDAQ + S&P 500.
"""
import os
import sys
import time

import trend_extreme_scan as tex

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

NEAR_PCT = 0.1   # within this % of today's high/low


def scan_at_extreme(tickers, tz):
    data = tex.fetch_15m(tickers)
    at_high, at_low = [], []
    scanned = 0
    for t, df in data.items():
        try:
            if df.index.tz is not None:
                df = df.tz_convert(tz)
            df = df.dropna(subset=["Close", "High", "Low"])
            if df.empty:
                continue
            today_date = df.index[-1].date()
            today_bars = df[df.index.date == today_date]
            if len(today_bars) < 3:
                continue
            scanned += 1
            day_high = float(today_bars["High"].max())
            day_low = float(today_bars["Low"].min())
            ltp = float(today_bars["Close"].iloc[-1])
            if day_high <= 0 or day_low <= 0:
                continue
            dist_high = (day_high - ltp) / day_high * 100
            dist_low = (ltp - day_low) / day_low * 100
            if dist_high <= NEAR_PCT:
                at_high.append((t.replace(".NS", ""), ltp))
            elif dist_low <= NEAR_PCT:
                at_low.append((t.replace(".NS", ""), ltp))
        except Exception:
            continue
    return at_high, at_low, scanned


def format_message(title, at_high, at_low, scanned, cur):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"<b>🎯 {title} — At Day High / Day Low</b>", f"{now} · {scanned} scanned",
             f"<i>Within {NEAR_PCT}% of today's extreme</i>", ""]
    lines.append(f"<b>🔼 At Day High ({len(at_high)}):</b>")
    for s, p in at_high[:30]:
        lines.append(f"  {s}  {cur}{p}")
    lines.append(f"\n<b>🔽 At Day Low ({len(at_low)}):</b>")
    for s, p in at_low[:30]:
        lines.append(f"  {s}  {cur}{p}")
    return "\n".join(lines)


def run_nse():
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        at_high, at_low, scanned = scan_at_extreme(tickers, "Asia/Kolkata")
        print(f"{name}: scanned={scanned} at_high={len(at_high)} at_low={len(at_low)}")
        msg = format_message(name, at_high, at_low, scanned, "₹")
        r = tex.send_telegram(msg)
        print("  sent:", r.get("ok"))
        time.sleep(1)


def run_nasdaq():
    from breakout_scan import load_universe
    tickers = load_universe()
    print(f"Scanning {len(tickers)} NASDAQ+S&P stocks...")
    at_high, at_low, scanned = scan_at_extreme(tickers, "America/New_York")
    print(f"scanned={scanned} at_high={len(at_high)} at_low={len(at_low)}")
    msg = format_message("NASDAQ + S&P 500", at_high, at_low, scanned, "$")
    r = tex.send_telegram(msg)
    print("sent:", r.get("ok"))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nasdaq":
        run_nasdaq()
    else:
        run_nse()
