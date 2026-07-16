"""
Telegram on-demand trigger for the breakout scanners (NASDAQ + NSE).
=======================================================================
Runs every 5 minutes (see .github/workflows/listen.yml). Each run:
  1. Checks Telegram for any new messages since the last check.
  2. "scan" / "run" / "results" / "update"      -> NASDAQ breakout scan
     "nifty" / "nse" / "india"                  -> NSE breakout scan (4 indices)
     "all"                                       -> both
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
ALL_WORDS    = {"all", "/all"}


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
    print("Running on-demand NASDAQ breakout scan...")
    tickers = load_universe()
    breakouts, scanned, errors = scan_breakouts(tickers)
    msg = "<b>📲 On-demand NASDAQ scan</b>\n" + format_telegram_message(breakouts, scanned, errors)
    result = send_telegram(msg)
    print("  Telegram send result:", result.get("ok"))


def run_nse_scan():
    print("Running on-demand NSE breakout scan...")
    indices = nse.load_indices()
    for key in ["nifty50", "nifty_next50", "midcap50", "smallcap250"]:
        tickers = indices[key]
        apply_ema = (key == "smallcap250")
        breakouts, scanned, errors = nse.scan_breakouts(tickers, apply_ema_filter=apply_ema)
        msg = "<b>📲 On-demand scan</b>\n" + nse.format_message(key, breakouts, scanned, apply_ema)
        result = nse.send_telegram(msg)
        print(f"  {key}: Telegram send result:", result.get("ok"))
        time.sleep(1)


def main():
    offset = get_offset()
    print(f"Checking Telegram for messages after update_id {offset}...")

    data = get_updates(offset + 1)
    updates = data.get("result", [])
    print(f"Found {len(updates)} new update(s).")

    if not updates:
        return

    max_update_id = offset
    want_nasdaq, want_nse = False, False

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
        elif text in ALL_WORDS:
            want_nasdaq = want_nse = True
            print(f"  ALL trigger matched: '{text}'")

    save_offset(max_update_id)

    if not (want_nasdaq or want_nse):
        print("No trigger word found in new messages — nothing to do.")
        return

    if want_nasdaq:
        run_nasdaq_scan()
    if want_nse:
        run_nse_scan()


if __name__ == "__main__":
    main()
