#!/usr/bin/env python3
"""Send a short note to the user's Telegram (used by the daily Claude review agent). Credentials come from the
live-alert launchd plist, never printed.  Usage: python3 send_note.py "message text"   (plain text)"""
import os
import plistlib
import sys

PLIST = os.path.expanduser("~/Library/LaunchAgents/com.ramandeep.livecandlealert.plist")
env = plistlib.load(open(PLIST, "rb")).get("EnvironmentVariables", {})
os.environ.setdefault("TELEGRAM_BOT_TOKEN", env["TELEGRAM_BOT_TOKEN"])
os.environ.setdefault("TELEGRAM_CHAT_ID", env["TELEGRAM_CHAT_ID"])
import live_candle_alert as L  # noqa: E402

text = " ".join(sys.argv[1:]).strip()
if not text:
    sys.exit("no text")
if os.path.exists(L.ALERTS_OFF_FILE):
    sys.exit("alerts are off (learning/ALERTS_OFF exists) - note not sent")
# plain text on purpose: agent notes often contain < > characters that break Telegram's HTML mode
import urllib.parse, urllib.request  # noqa: E402,E401
url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
for i in range(0, len(text), 3800):
    data = urllib.parse.urlencode({"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text[i:i + 3800]}).encode()
    urllib.request.urlopen(urllib.request.Request(url, data=data, method="POST"), timeout=15)
print("sent", len(text), "chars")
