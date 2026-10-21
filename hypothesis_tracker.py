#!/usr/bin/env python3
"""Daily test of the trend-hold hypothesis.

HYPOTHESIS (user, 2026-10-21): among stocks in the 3 PM trend-hold scan, the ones CLOSEST to their EMA50 move
more in the scan's direction on the next trading day than the ones farther away (bearish list: they fall more;
bullish list, mirrored: they rise more). The same split is also run on distance to the EMA100.

What this script does each run (idempotent):
  1. Downloads 60 days of 15m data for the F&O stocks and Nifty.
  2. Seeds learning/trend_scan_log.csv with a replay of the scan over past days (source=backfill) so the test has history.
  3. Measures every logged scan whose NEXT trading day has completed: scan-day close -> next-day close, signed in the scan
     direction, minus Nifty's move in the same direction (learning/hypothesis_results.csv).
  4. Per day and side: closest half vs farthest half; accumulates mean difference, t-stat, % of days confirming.
     Live days (source=live) are the genuine out-of-sample test; backfill days are the in-sample baseline.
  5. Writes learning/hypothesis_summary.md and sends a short Telegram update when a new live day was measured.

Usage: python3 hypothesis_tracker.py [--dry] [--force]
"""
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live_candle_alert as L
import trend_hold_scan as T

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "learning", "hypothesis_results.csv")
SUMMARY = os.path.join(BASE, "learning", "hypothesis_summary.md")
MARK = os.path.join(BASE, "learning", "hypothesis_sent.json")
TZ = "Asia/Kolkata"
MIN_STOCKS = 6          # per side and day, otherwise the half-vs-half split is meaningless


def session_table(b):
    """Per day: first open, last close, and whether the 15:15 candle exists (complete session)."""
    g = b.groupby("day")
    t = pd.DataFrame({"open": g.Open.first(), "close": g.Close.last(), "last": g.apply(lambda x: x.index[-1].strftime("%H:%M"))})
    t["complete"] = t["last"] >= "15:15"
    return t


def seed_log(data):
    R = T.backfill(data)
    if R.empty:
        return
    R["source"] = "backfill"
    old = pd.read_csv(T.LOG) if os.path.exists(T.LOG) else pd.DataFrame()
    if len(old):
        live = old[old.source == "live"]
        live_days = set(live.day)
        R = R[~R.day.isin(live_days)]
        R = pd.concat([live, R], ignore_index=True)
    R.to_csv(T.LOG, index=False)


def measure(data, nifty):
    log = pd.read_csv(T.LOG)
    sess = {}
    for t, df in data.items():
        b = T.prep(df)
        if b is not None:
            sess[t] = session_table(b)
    nb = nifty.copy()
    nb.index = nb.index.tz_convert(TZ)
    nb["day"] = nb.index.strftime("%Y-%m-%d")
    ns = session_table(nb)
    done = pd.read_csv(RESULTS) if os.path.exists(RESULTS) else pd.DataFrame()
    have = set(zip(done.day, done.t)) if len(done) else set()
    rows = []
    for _, r in log.iterrows():
        if (r.day, r.t) in have or r.t not in sess:
            continue
        s = sess[r.t]
        days = list(s.index)
        if r.day not in days:
            continue
        k = days.index(r.day)
        if k + 1 >= len(days) or not s.complete.iloc[k + 1] or r.day not in ns.index or days[k + 1] not in ns.index:
            continue
        n = days[k + 1]
        sgn = -1 if r.side == "BEAR" else 1
        base = float(s.close.iloc[k])
        ret = sgn * (float(s.close.iloc[k + 1]) / base - 1) * 100
        nret = sgn * (float(ns.close[n]) / float(ns.close[r.day]) - 1) * 100
        rows.append(dict(day=r.day, next_day=n, t=r.t, side=r.side, d50=r.d50, d100=r.d100, ret=ret, nifty=nret, excess=ret - nret, source=r.source))
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(RESULTS, index=False)
    return done


def split_stats(done, col):
    """Per day and side: mean excess of the closest half minus the farthest half (positive = hypothesis confirmed)."""
    out = []
    for (day, side), g in done.groupby(["day", "side"]):
        if len(g) < MIN_STOCKS:
            continue
        d = g[col].abs()
        near, far = g[d <= d.median()], g[d > d.median()]
        if len(near) == 0 or len(far) == 0:
            continue
        out.append(dict(day=day, side=side, source=g.source.iloc[0], n=len(g), near=near.excess.mean(), far=far.excess.mean(),
                        diff=near.excess.mean() - far.excess.mean()))
    return pd.DataFrame(out)


def summarize(S):
    if S.empty:
        return "no data yet"
    n = len(S)
    m = S["diff"].mean()
    t = m / (S["diff"].std() / np.sqrt(n)) if n > 4 and S["diff"].std() > 0 else float("nan")
    return f"{n} days | mean(closest - farthest) {m:+.2f}% | confirmed on {(S['diff'] > 0).mean() * 100:.0f}% of days | t={t:+.2f}"


def report(done):
    lines = ["# Trend-hold hypothesis tracker",
             "Hypothesis: closest to EMA50 (and EMA100) moves MORE in the scan direction next day (bearish list: falls more; bullish list: rises more).",
             "Positive 'diff' = closest half beat the farthest half (hypothesis confirmed). Excess = move minus Nifty's move in the same direction.\n"]
    msg = []
    for col, lab in (("d50", "EMA50"), ("d100", "EMA100")):
        S = split_stats(done, col)
        lines.append(f"## Distance to {lab}")
        for side in ("BEAR", "BULL"):
            for src, srcname in (("backfill", "backfill (in-sample baseline)"), ("live", "LIVE (out-of-sample)")):
                Z = S[(S.side == side) & (S.source == src)] if len(S) else S
                lines.append(f"- {side} {srcname}: {summarize(Z)}")
        lines.append("")
    S50 = split_stats(done, "d50")
    last_live = S50[S50.source == "live"].sort_values("day").tail(2) if len(S50) else S50
    for _, r in last_live.iterrows():
        msg.append(f"{r.day} {r.side} ({int(r.n)} stocks): closest-to-EMA50 half {r.near:+.2f}% vs farthest half {r.far:+.2f}% (vs Nifty) -> "
                   f"{'CONFIRMED' if r['diff'] > 0 else 'not confirmed'} ({r['diff']:+.2f}%)")
    return "\n".join(lines), msg, S50


SKILL_DIR = os.path.expanduser("~/.claude/skills/trend-hold-hypothesis")
LEARN = os.path.join(SKILL_DIR, "LEARNINGS.md")
HIST = os.path.join(BASE, "learning", "hypothesis_history.csv")
A0, A1 = "<!-- AUTO:START -->", "<!-- AUTO:END -->"


def write_learnings(done, text):
    """Rewrite the AUTO block of the skill's LEARNINGS.md with the latest facts; the agent's own notes below it are kept."""
    S50, S100 = split_stats(done, "d50"), split_stats(done, "d100")
    today = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d")
    row = dict(date=today)
    for side in ("BEAR", "BULL"):
        for src in ("backfill", "live"):
            for nm, S in (("e50", S50), ("e100", S100)):
                Z = S[(S.side == side) & (S.source == src)] if len(S) else S
                row[f"{side}_{src}_{nm}_days"] = len(Z)
                row[f"{side}_{src}_{nm}_diff"] = round(Z["diff"].mean(), 3) if len(Z) else np.nan
                row[f"{side}_{src}_{nm}_conf"] = round((Z["diff"] > 0).mean() * 100, 1) if len(Z) else np.nan
    h = pd.read_csv(HIST) if os.path.exists(HIST) else pd.DataFrame()
    h = pd.concat([h[h.date != today] if len(h) else h, pd.DataFrame([row])], ignore_index=True)
    h.to_csv(HIST, index=False)
    health = "unknown"
    try:
        lines = [x for x in open(os.path.expanduser("~/Library/Logs/live_candle_alert.log")) if "NSE:" in x and x.startswith("[")]
        health = lines[-1].strip()[:140]
    except Exception:
        pass
    sc = ""
    try:
        card = json.load(open(os.path.join(BASE, "learning", "scorecard.json")))
        sc = f"Alert-setup scorecard: {card.get('signals', '?')} signals over {card.get('days', '?')} days through {card.get('through', '?')}, proven setups: {len(card.get('proven', []))}."
    except Exception:
        pass
    block = [A0, f"## Auto-updated facts ({datetime.now(ZoneInfo(TZ)).strftime('%Y-%m-%d %H:%M')} IST, written by hypothesis_tracker.py - do not edit by hand)", "",
             text.split("\n", 3)[-1].strip(), "", sc, f"Last live NSE scan cycle: {health}", "",
             "### History (last 20 updates; bear = closest-minus-farthest next-day excess %, live = out-of-sample)", "",
             "| date | live days (EMA50) | bear live diff e50 | bear live diff e100 | bear baseline e100 | bull live diff e50 |", "|---|---|---|---|---|---|"]
    for _, r in h.tail(20).iterrows():
        block.append(f"| {r.date} | {int(r.get('BEAR_live_e50_days', 0))} | {r.get('BEAR_live_e50_diff', '')} | {r.get('BEAR_live_e100_diff', '')} | {r.get('BEAR_backfill_e100_diff', '')} | {r.get('BULL_live_e50_diff', '')} |")
    block.append(A1)
    new = "\n".join(block)
    os.makedirs(SKILL_DIR, exist_ok=True)
    old = open(LEARN).read() if os.path.exists(LEARN) else ""
    if A0 in old and A1 in old:
        out = old[:old.index(A0)] + new + old[old.index(A1) + len(A1):]
    else:
        out = ("# Trend-hold hypothesis - learnings\n\nThe block between the AUTO markers is rewritten every day by hypothesis_tracker.py. "
               "Everything below it belongs to the daily Claude review agent (notes and proposals).\n\n" + new +
               "\n\n## Agent review log (newest first)\n\n(none yet)\n\n## Proposals awaiting the user's approval\n\n(none)\n")
    open(LEARN, "w").write(out)


def main():
    dry, force = "--dry" in sys.argv, "--force" in sys.argv
    tick = T.universe()
    data = L.fetch(tick + ["^NSEI"], period="60d")
    nifty = data.pop("^NSEI")
    nifty = nifty.dropna(subset=["Open", "High", "Low", "Close"])
    seed_log(data)
    done = measure(data, nifty)
    if done.empty:
        print("no measurable scans yet")
        return
    text, msg, S50 = report(done)
    open(SUMMARY, "w").write(text)
    write_learnings(done, text)
    print(text)
    live_days = sorted(set(S50[S50.source == "live"].day)) if len(S50) else []
    try:
        mark = json.load(open(MARK))
    except Exception:
        mark = {}
    newest = live_days[-1] if live_days else None
    if newest and (mark.get("last_live_day") != newest or force):
        head = "<b>Hypothesis test: closer to EMA50 moves more the next day?</b>"
        body = "\n".join(msg)
        cum = []
        for side in ("BEAR", "BULL"):
            Z = S50[(S50.side == side) & (S50.source == "live")]
            cum.append(f"LIVE {side}: {summarize(Z)}")
            Z = S50[(S50.side == side) & (S50.source == "backfill")]
            cum.append(f"baseline {side}: {summarize(Z)}")
        L.send(head + "\n" + body + "\n\n" + "\n".join(cum), dry)
        if not dry:
            json.dump({"last_live_day": newest}, open(MARK, "w"))


if __name__ == "__main__":
    L.raise_fd_limit()
    main()
