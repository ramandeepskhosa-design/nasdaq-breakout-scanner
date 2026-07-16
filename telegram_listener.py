"""
Telegram on-demand trigger for the NASDAQ breakout scanner.
=============================================================
Runs a few times a day (see .github/workflows/listen.yml). Each run:
  1. Checks Telegram for any new messages since the last check.
  2. If any message says "scan" / "run" (case-insensitive), runs the
     full breakout scan and replies with results.
  3. Remembers which messages it already saw (offset.txt) so it never
     double-replies, and commits that file back to the repo.

This is polling, not a live webhook — replies land at the next
scheduled check after you send your message, not instantly.
"""
import json
import os
import urllib.request
import urllib.parse

from breakout_scan import (
    load_universe, scan_breakouts, format_telegram_message, send_telegram,
)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OFFSET_FILE = os.path.join(BASE_DIR, "offset.txt")

TRIGGER_WORDS = {"scan", "run", "/scan", "/run", "results", "update"}


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


def main():
    offset = get_offset()
    print(f"Checking Telegram for messages after update_id {offset}...")

    data = get_updates(offset + 1)
    updates = data.get("result", [])
    print(f"Found {len(updates)} new update(s).")

    if not updates:
        return

    max_update_id = offset
    should_scan = False

    for u in updates:
        max_update_id = max(max_update_id, u["update_id"])
        msg = u.get("message", {})
        text = (msg.get("text") or "").strip().lower()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != TELEGRAM_CHAT_ID:
            continue   # ignore messages from anyone else
        if text in TRIGGER_WORDS:
            should_scan = True
            print(f"  Trigger word matched: '{text}'")

    save_offset(max_update_id)

    if not should_scan:
        print("No trigger word found in new messages — nothing to do.")
        return

    print("Running on-demand breakout scan...")
    tickers = load_universe()
    breakouts, scanned, errors = scan_breakouts(tickers)
    msg = "<b>📲 On-demand scan (you asked)</b>\n" + format_telegram_message(breakouts, scanned, errors)
    result = send_telegram(msg)
    print("Telegram send result:", result.get("ok"))


if __name__ == "__main__":
    main()
