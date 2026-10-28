#!/usr/bin/env python3
"""What has the trading system NOT done yet for the last completed NSE session?

Checks (for the latest complete session D):
  * live daily signal report sent        * nightly learning report sent
  * 3 PM trend-hold scan sent            * hypothesis tracker ran
  * daily Claude review agent ran        * live alert service running
Prints a short list with the fix for each item.

Modes:
  (default)   print the list
  --context   plain text for a Claude session start (SessionStart hook): says what is pending and tells Claude to ask the user
  --notify    macOS notification + Telegram note when something is pending (at most once every 4 hours); used by launchd
  --fix       run the safe catch-ups (nightly learning, hypothesis tracker) - never restarts services
"""
import json
import os
import plistlib
import re
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(BASE, "learning")
NOTE_STATE = os.path.join(LEARN, "pending_notified.json")
SKILL_LEARN = os.path.expanduser("~/.claude/skills/trend-hold-hypothesis/LEARNINGS.md")
TZ = ZoneInfo("Asia/Kolkata")
PYTHON = "/opt/homebrew/bin/python3"


def last_complete_session():
    """Date of the latest NSE session whose 15:15 candle exists (Yahoo ^NSEI 15m). None if it cannot be checked."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", period="10d", interval="15m", progress=False, auto_adjust=False)
        if df is None or len(df) == 0:
            return None
        df = df.dropna(how="all")
        idx = df.index.tz_convert("Asia/Kolkata")
        days = {}
        for ts in idx:
            days[ts.strftime("%Y-%m-%d")] = ts.strftime("%H:%M")
        done = [d for d, last in days.items() if last >= "15:15"]
        return max(done) if done else None
    except Exception as e:
        print("session check failed:", str(e)[:100], file=sys.stderr)
        return None


def read(path, default=""):
    try:
        return open(path).read().strip()
    except Exception:
        return default


def service_running(label):
    try:
        out = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/{label}"], capture_output=True, text=True, timeout=10).stdout
        return "state = running" in out
    except Exception:
        return False


def review_done_day():
    """Latest date mentioned in the agent review log section of LEARNINGS.md (the agent writes dated entries)."""
    txt = read(SKILL_LEARN)
    if "## Agent review log" not in txt:
        return ""
    sec = txt.split("## Agent review log", 1)[1].split("## Proposals", 1)[0]
    dates = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", sec)
    marker = read(os.path.join(LEARN, "review_done.json"))
    try:
        dates.append(json.loads(marker).get("day", ""))
    except Exception:
        pass
    return max(dates) if dates else ""


def pending():
    D = last_complete_session()
    items = []
    if not D:
        return None, [("could not check (offline or Yahoo unavailable)", "")]
    now = datetime.now(TZ)
    today = now.strftime("%Y-%m-%d")

    def due(hhmm):
        """For today's session, an item only counts as pending once its scheduled time (+ grace) has passed."""
        return D != today or now.strftime("%H:%M") >= hhmm

    # nightly learning report
    if read(os.path.join(LEARN, "last_sent.txt")) < D and due("16:15"):
        items.append((f"nightly learning report for {D} not sent", f"cd {BASE} && {PYTHON} signal_learning.py"))
    # 3 PM trend-hold scan (cannot be re-run for a past day: it needs the 15:00 candle)
    try:
        ts = json.load(open(os.path.join(LEARN, "trend_scan_sent.json"))).get("day", "")
    except Exception:
        ts = ""
    if ts < D and due("15:40"):
        items.append((f"3 PM trend-hold scan for {D} was missed (Mac off at 15:01); it cannot be replayed live - tomorrow's run continues the log", ""))
    # hypothesis tracker
    hist = read(os.path.join(LEARN, "hypothesis_history.csv"))
    last_hist = hist.splitlines()[-1].split(",")[0] if hist.count("\n") >= 1 else ""
    if last_hist < D and due("16:30"):
        items.append((f"hypothesis tracker has not measured {D} yet", f"cd {BASE} && {PYTHON} hypothesis_tracker.py"))
    # daily review agent
    if review_done_day() < D and due("17:30"):
        items.append((f"daily Claude review for {D} has not run (it runs when the app is open at ~16:30, or run 'trading-bot-daily-review' from the Scheduled section)", ""))
    # live daily signal report
    try:
        st = json.load(open(os.path.join(BASE, ".live_alert_state.json")))
        if f"REPORT|NSE|{D}" not in st and due("15:50"):
            items.append((f"live NSE daily signal report for {D} not sent (the live service sends it when it next runs)", ""))
    except Exception:
        pass
    # intraday 09:30 -> 10:30 morning screen
    try:
        ms = json.load(open(os.path.join(LEARN, "morning_sent.json"))).get("day", "")
    except Exception:
        ms = ""
    if ms and ms < D and due("09:45"):
        items.append((f"the 09:30 morning list for {D} was missed (Mac off at 09:32); it cannot be replayed live - tomorrow's run continues", ""))
    try:
        mt = json.load(open(os.path.join(LEARN, "morning_tracked.json"))).get("day", "")
    except Exception:
        mt = ""
    if ms >= D and mt < D and due("10:50"):
        items.append((f"09:30->10:30 results for {D} not measured yet", f"cd {BASE} && {PYTHON} intraday_0930.py track"))
    # live service
    if not service_running("com.ramandeep.livecandlealert"):
        items.append(("live alert service is NOT running", f"launchctl kickstart -k gui/{os.getuid()}/com.ramandeep.livecandlealert"))
    return D, items


def notify(items, D):
    if os.path.exists(os.path.join(LEARN, "ALERTS_OFF")):
        return
    try:
        st = json.load(open(NOTE_STATE))
    except Exception:
        st = {}
    sig = "|".join(sorted(i[0] for i in items))
    if st.get("sig") == sig and time.time() - st.get("t", 0) < 4 * 3600:
        return
    text = f"Trading bot: {len(items)} item(s) pending for {D}:\n" + "\n".join(f"- {i[0]}" for i in items) + "\nOpen Claude and say 'catch up' to fix them."
    short = f"{len(items)} pending: " + "; ".join(i[0][:40] for i in items[:2])
    subprocess.run(["osascript", "-e", f'display notification "{short}" with title "Trading bot" subtitle "pending for {D}"'], timeout=10)
    try:
        env = plistlib.load(open(os.path.expanduser("~/Library/LaunchAgents/com.ramandeep.livecandlealert.plist"), "rb"))["EnvironmentVariables"]
        import urllib.parse
        import urllib.request
        data = urllib.parse.urlencode({"chat_id": env["TELEGRAM_CHAT_ID"], "text": text}).encode()
        urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{env['TELEGRAM_BOT_TOKEN']}/sendMessage", data=data, method="POST"), timeout=15)
    except Exception as e:
        print("telegram note failed:", e)
    os.makedirs(LEARN, exist_ok=True)
    json.dump({"sig": sig, "t": time.time()}, open(NOTE_STATE, "w"))


def main():
    D, items = pending()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if not items:
        if mode == "--context":
            print(f"[trading-bot] Nothing pending for the last NSE session ({D}).")
        elif mode == "":
            print(f"Nothing pending for the last complete NSE session ({D}).")
        return
    if mode == "--context":
        print(f"[trading-bot] PENDING for the last NSE session ({D}):")
        for t, fix in items:
            print(f"  - {t}" + (f"  | fix: {fix}" if fix else ""))
        print("INSTRUCTION: in your first reply, briefly tell the user these items are pending and ask whether to run the fixes now "
              "(the user's trading-alert system; do not run anything before they say yes). If they say 'catch up', run the listed fix commands "
              "(never restart services during NSE hours 09:15-15:30 IST without asking), then summarize.")
    elif mode == "--notify":
        notify(items, D)
    elif mode == "--fix":
        for t, fix in items:
            if fix and "kickstart" not in fix:
                print("running:", fix)
                subprocess.run(fix, shell=True)
    else:
        print(f"Pending for {D}:")
        for t, fix in items:
            print(f" - {t}" + (f"\n     fix: {fix}" if fix else ""))


if __name__ == "__main__":
    main()
