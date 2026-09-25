"""
Telegram on-demand trigger for the breakout scanners (NASDAQ + NSE).
=======================================================================
Runs every 5 minutes (see .github/workflows/listen.yml). Each run:
  1. Checks Telegram for any new messages since the last check.
  2. "scan" / "run" / "results" / "update"      -> NASDAQ breakout scan
     "nifty" / "nse" / "india"                  -> NSE breakout scan (4 indices)
     "wick" / "nowick" / "gap"                  -> EOD gap-up/no-lower-wick scan (full day)
     "openwick" / "earlywick"                   -> opening-candle gap-up/no-wick scan (first 15 min)
     "all"                                       -> everything
  3. Remembers which messages it already saw (offset.txt) so it never
     double-replies, and commits that file back to the repo.

This is polling, not a live webhook — replies land at the next
scheduled check after you send your message, not instantly.
"""
import json
import os
import time
import urllib.request
import urllib.parse

from breakout_scan import (
    load_universe, scan_breakouts, format_telegram_message, send_telegram,
)
import nse_breakout_scan as nse

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OFFSET_FILE = os.path.join(BASE_DIR, "offset.txt")

NASDAQ_WORDS = {"scan", "run", "/scan", "/run", "results", "update"}
NSE_WORDS    = {"nifty", "nse", "india", "/nifty", "/nse"}
WICK_WORDS      = {"wick", "nowick", "gap", "/wick", "/gap"}
OPEN_WICK_WORDS = {"openwick", "earlywick", "/openwick", "/earlywick"}
EMA20_WORDS     = {"ema20", "hold", "consolidate", "flag", "/ema20", "/hold"}
SMOOTH_EMA_WORDS = {"smooth", "parallel", "coil", "/smooth", "/coil"}
OPTIONS_WORDS    = {"options", "atm", "crossover", "/options", "/atm"}
TREND_WORDS       = {"trend", "extreme", "smoothtrend", "/trend"}
DAY_EXTREME_WORDS = {"dayhigh", "daylow", "highlow", "/dayhigh", "/daylow"}
CRYPTO_WORDS      = {"crypto", "bitcoin", "btc", "/crypto", "/btc"}
INSTITUTIONAL_WORDS = {"institutional", "fullbody", "conviction", "structure", "/institutional"}
ALL_WORDS       = {"all", "/all"}


def get_offset():
    if os.path.exists(OFFSET_FILE):
        with open(OFFSET_FILE) as f:
            return int(f.read().strip() or 0)
    return 0


def save_offset(update_id):
    with open(OFFSET_FILE, "w") as f:
        f.write(str(update_id))


def get_updates(offset):
    url = (f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
           f"?offset={offset}&timeout=5")
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


def run_nasdaq_scan():
    print("Running on-demand NASDAQ signal scan (RSI+EMA stack)...")
    tickers = load_universe()
    from breakout_scan import scan_signal, format_signal_message
    results, scanned, errors = scan_signal(tickers)
    msg = "<b>📲 On-demand scan</b>\n" + format_signal_message(results, scanned)
    result = send_telegram(msg)
    print("  Telegram send result:", result.get("ok"))


def run_nse_scan():
    print("Running on-demand NSE signal scan (RSI+EMA stack)...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        use_lifetime = (key == "smallcap250")
        results, scanned, errors = nse.scan_signal(tickers, use_lifetime_high=use_lifetime)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_signal_message(key, results, scanned, use_lifetime_high=use_lifetime)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)


def run_wick_scan():
    print("Running on-demand gap-up/no-lower-wick scan (NSE)...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        results, scanned, errors = nse.scan_no_wick_gap(tickers)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_no_wick_message(key, results, scanned)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand gap-up/no-lower-wick scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    from breakout_scan import scan_no_wick_gap, format_no_wick_message
    results, scanned, errors = scan_no_wick_gap(tickers)
    msg = "<b>📲 On-demand scan</b>\n" + format_no_wick_message(results, scanned)
    result = send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_open_wick_scan():
    print("Running on-demand opening-candle gap-up/no-wick scan (NSE)...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        results, scanned, errors = nse.scan_opening_wick(tickers)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_opening_wick_message(key, results, scanned)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand opening-candle gap-up/no-wick scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    from breakout_scan import scan_opening_wick, format_opening_wick_message
    results, scanned, errors = scan_opening_wick(tickers)
    msg = "<b>📲 On-demand scan</b>\n" + format_opening_wick_message(results, scanned)
    result = send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_ema20_hold_scan():
    print("Running on-demand EMA20 hold consolidation scan (NSE)...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        results, scanned, errors = nse.scan_ema20_hold(tickers)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_ema20_hold_message(key, results, scanned)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand EMA20 hold consolidation scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    from breakout_scan import scan_ema20_hold, format_ema20_hold_message
    results, scanned, errors = scan_ema20_hold(tickers)
    msg = "<b>📲 On-demand scan</b>\n" + format_ema20_hold_message(results, scanned)
    result = send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_smooth_ema_scan():
    print("Running on-demand smooth/parallel EMA5-EMA9 scan (NSE)...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        results, scanned, errors = nse.scan_smooth_parallel_ema(tickers)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_smooth_parallel_ema_message(key, results, scanned)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand smooth/parallel EMA5-EMA9 scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    from breakout_scan import scan_smooth_parallel_ema, format_smooth_parallel_ema_message
    results, scanned, errors = scan_smooth_parallel_ema(tickers)
    msg = "<b>📲 On-demand scan</b>\n" + format_smooth_parallel_ema_message(results, scanned)
    result = send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_options_ema_scan():
    print("Running on-demand options ATM crossover scan (NIFTY/BANKNIFTY/F&O stocks)...")
    import options_ema_scan as opt
    results, scanned, errors = opt.scan_ema_crossovers()
    msg = "<b>📲 On-demand scan</b>\n" + opt.format_message(results, scanned)
    result = opt.send_telegram(msg)
    print("  Telegram send result:", result.get("ok"))


def run_trend_extreme_scan():
    print("Running on-demand trending-at-extreme scan (NSE)...")
    import trend_extreme_scan as tex
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        results, scanned, errors = tex.scan_trending_at_extreme(tickers, tz="Asia/Kolkata")
        for r in results:
            r["sym"] = r["sym"].replace(".NS", "")
        msg = "<b>📲 On-demand scan</b>\n" + tex.format_message(name, results, scanned, currency="₹")
        result = tex.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand trending-at-extreme scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    results, scanned, errors = tex.scan_trending_at_extreme(tickers, tz="America/New_York")
    msg = "<b>📲 On-demand scan</b>\n" + tex.format_message("NASDAQ + S&P 500", results, scanned, currency="$")
    result = tex.send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_day_extreme_scan():
    print("Running on-demand at-day-high/day-low scan (NSE)...")
    import day_extreme_scan as dex
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        at_high, at_low, scanned = dex.scan_at_extreme(tickers, "Asia/Kolkata")
        msg = "<b>📲 On-demand scan</b>\n" + dex.format_message(name, at_high, at_low, scanned, "₹")
        result = dex.tex.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand at-day-high/day-low scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    at_high, at_low, scanned = dex.scan_at_extreme(tickers, "America/New_York")
    msg = "<b>📲 On-demand scan</b>\n" + dex.format_message("NASDAQ + S&P 500", at_high, at_low, scanned, "$")
    result = dex.tex.send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def run_crypto_scan():
    print("Running on-demand crypto scan (BTC/ETH/SOL/BNB/XRP/DOGE)...")
    import crypto_scan as crypto
    rows = crypto.analyze()
    msg = "<b>📲 On-demand scan</b>\n" + crypto.format_message(rows)
    result = crypto.send_telegram(msg)
    print("  Telegram send result:", result.get("ok"))


def run_institutional_trend_scan():
    print("Running on-demand institutional trend (full-body/low-wick) scan (NSE)...")
    import institutional_trend_scan as its
    import nse_breakout_scan as nse
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = [f"{s}.NS" for s in indices[key]]
        name = nse.INDEX_NAMES[key]
        results, scanned, errors = its.scan_institutional_trend(tickers, "Asia/Kolkata")
        msg = "<b>📲 On-demand scan</b>\n" + its.format_message(name, results, scanned, "₹")
        result = its.tex.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)

    print("Running on-demand institutional trend scan (NASDAQ + S&P 500)...")
    tickers = load_universe()
    results, scanned, errors = its.scan_institutional_trend(tickers, "America/New_York")
    msg = "<b>📲 On-demand scan</b>\n" + its.format_message("NASDAQ + S&P 500", results, scanned, "$")
    result = its.tex.send_telegram(msg)
    print("  NASDAQ+S&P: Telegram send result:", result.get("ok"))


def main():
    offset = get_offset()
    print(f"Checking Telegram for messages after update_id {offset}...")

    data = get_updates(offset + 1)
    updates = data.get("result", [])
    print(f"Found {len(updates)} new update(s).")

    if not updates:
        return

    max_update_id = offset
    want_nasdaq, want_nse, want_wick, want_open_wick, want_ema20, want_smooth, want_options, want_trend, want_dayext, want_crypto, want_institutional = \
        False, False, False, False, False, False, False, False, False, False, False

    for u in updates:
        max_update_id = max(max_update_id, u["update_id"])
        msg = u.get("message", {})
        text = (msg.get("text") or "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != TELEGRAM_CHAT_ID:
            continue   # ignore messages from anyone else
        if text in NASDAQ_WORDS:
            want_nasdaq = True
            print(f"  NASDAQ trigger matched: '{text}'")
        elif text in NSE_WORDS:
            want_nse = True
            print(f"  NSE trigger matched: '{text}'")
        elif text in INSTITUTIONAL_WORDS:
            want_institutional = True
            print(f"  INSTITUTIONAL TREND trigger matched: '{text}'")
        elif text in CRYPTO_WORDS:
            want_crypto = True
            print(f"  CRYPTO trigger matched: '{text}'")
        elif text in DAY_EXTREME_WORDS:
            want_dayext = True
            print(f"  DAY HIGH/LOW trigger matched: '{text}'")
        elif text in TREND_WORDS:
            want_trend = True
            print(f"  TREND EXTREME trigger matched: '{text}'")
        elif text in OPTIONS_WORDS:
            want_options = True
            print(f"  OPTIONS trigger matched: '{text}'")
        elif text in SMOOTH_EMA_WORDS:
            want_smooth = True
            print(f"  SMOOTH EMA trigger matched: '{text}'")
        elif text in EMA20_WORDS:
            want_ema20 = True
            print(f"  EMA20 HOLD trigger matched: '{text}'")
        elif text in OPEN_WICK_WORDS:
            want_open_wick = True
            print(f"  OPEN WICK trigger matched: '{text}'")
        elif text in WICK_WORDS:
            want_wick = True
            print(f"  WICK trigger matched: '{text}'")
        elif text in ALL_WORDS:
            want_nasdaq = want_nse = want_wick = want_open_wick = want_ema20 = want_smooth = want_options = want_trend = want_dayext = want_crypto = want_institutional = True
            print(f"  ALL trigger matched: '{text}'")

    save_offset(max_update_id)

    if not (want_nasdaq or want_nse or want_wick or want_open_wick or want_ema20 or want_smooth or want_options or want_trend or want_dayext or want_crypto or want_institutional):
        print("No trigger word found in new messages — nothing to do.")
        return

    if want_nasdaq:
        run_nasdaq_scan()
    if want_options:
        run_options_ema_scan()
    if want_trend:
        run_trend_extreme_scan()
    if want_dayext:
        run_day_extreme_scan()
    if want_crypto:
        run_crypto_scan()
    if want_institutional:
        run_institutional_trend_scan()
    if want_open_wick:
        run_open_wick_scan()
    if want_ema20:
        run_ema20_hold_scan()
    if want_smooth:
        run_smooth_ema_scan()
    if want_nse:
        run_nse_scan()
    if want_wick:
        run_wick_scan()


if __name__ == "__main__":
    main()
