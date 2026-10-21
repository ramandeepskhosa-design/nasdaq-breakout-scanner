#!/usr/bin/env python3
"""3 PM trend-hold scan - NSE F&O stocks only.

BEARISH: for the last WINDOW 15m candles the price has stayed below the EMA20 (the allowed relaxation),
         with at least MIN_SHARE of those candles closing below BOTH EMA5 and EMA9, the latest close is below
         EMA5 and EMA9, and the slow EMAs are descending: EMA50 < EMA100 < EMA200.
BULLISH: the mirror image (above EMA20, mostly above EMA5/EMA9, latest close above both, EMA50 > EMA100 > EMA200).
Lists are sorted with the stock CLOSEST to its EMA100 first (smallest distance, ascending).

Every live run is also appended to learning/trend_scan_log.csv; hypothesis_tracker.py then measures the next day
(hypothesis: stocks CLOSEST to the EMA50 fall more the next day than those farther away).

Runs once a day at ~15:01 IST (launchd), idempotent via learning/trend_scan_sent.json.
Usage: python3 trend_hold_scan.py [--dry] [--force] [--window=12] [--share=0.75]
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live_candle_alert as L

BASE = os.path.dirname(os.path.abspath(__file__))
SENT = os.path.join(BASE, "learning", "trend_scan_sent.json")
LOG = os.path.join(BASE, "learning", "trend_scan_log.csv")
TZ = "Asia/Kolkata"
DEAD = {"TATAMOTORS", "LTIM", "ZOMATO", "MCDOWELL-N"}
MAX_PER_SIDE = 30
WINDOW, SHARE = 12, 0.75


def universe():
    return [s + ".NS" for s in L.fo_symbols() if s not in DEAD]


def prep(df, now=None):
    b = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if now is not None:
        b = L.completed(b, now).copy()
    if len(b) < 260:
        return None
    b.index = b.index.tz_convert(TZ)
    b["day"] = b.index.strftime("%Y-%m-%d")
    for p in (5, 9, 20, 50, 100, 200):
        b[f"e{p}"] = b.Close.ewm(span=p, adjust=False).mean()
    tp = (b.High + b.Low + b.Close) / 3
    b["vwap"] = (tp * b.Volume).groupby(b.day).cumsum() / b.Volume.groupby(b.day).cumsum()
    return b


def evaluate(b, i, window=WINDOW, share=SHARE):
    """Scan one stock at candle position i. Returns a dict or None."""
    if i < window + 1:
        return None
    sl = slice(i - window + 1, i + 1)
    c = b.Close.iloc[sl]
    e5, e9, e20 = b.e5.iloc[sl], b.e9.iloc[sl], b.e20.iloc[sl]
    below_both = int((c < np.minimum(e5, e9)).sum())
    above_both = int((c > np.maximum(e5, e9)).sum())
    r = b.iloc[i]
    px = float(r.Close)
    side = None
    if (c < e20).all() and below_both >= math.ceil(share * window) and px < min(r.e5, r.e9) and r.e50 < r.e100 < r.e200:
        side = "BEAR"
    elif (c > e20).all() and above_both >= math.ceil(share * window) and px > max(r.e5, r.e9) and r.e50 > r.e100 > r.e200:
        side = "BULL"
    if not side:
        return None
    day = b.day.iloc[i]
    prev = b[b.day < day]
    today = b[(b.day == day)].iloc[: (b.day.iloc[:i + 1] == day).sum()]
    vw = r.vwap
    return dict(side=side, px=px, d100=(px / r.e100 - 1) * 100, d50=(px / r.e50 - 1) * 100, d20=(px / r.e20 - 1) * 100,
                n_strict=below_both if side == "BEAR" else above_both,
                day_chg=(px / prev.Close.iloc[-1] - 1) * 100 if len(prev) else np.nan,
                vwap_side=("below" if px < vw else "above") if vw == vw else "n/a",
                vol=L.slot_volume_ratio(b.iloc[:i + 1], str(b.index[i])), bar=b.index[i].strftime("%H:%M"), day=day)


def scan_live(data, now):
    rows = []
    for t, df in data.items():
        b = prep(df, now)
        if b is None:
            continue
        r = evaluate(b, len(b) - 1)
        if r:
            rows.append(dict(t=t, **r))
    return pd.DataFrame(rows)


def backfill(data, days=None, window=WINDOW, share=SHARE):
    """Replay the scan at the 14:45 candle (known at 15:00) of every past day. Used to seed the hypothesis test."""
    rows = []
    for t, df in data.items():
        b = prep(df)
        if b is None:
            continue
        pos = np.where((b.index.hour == 14) & (b.index.minute == 45))[0]
        for i in pos:
            if days is not None and b.day.iloc[i] not in days:
                continue
            r = evaluate(b, i, window, share)
            if r:
                rows.append(dict(t=t, **r))
    return pd.DataFrame(rows)


def build_message(R, now, total, window=WINDOW):
    day = now.astimezone(ZoneInfo(TZ)).strftime("%Y-%m-%d %H:%M")
    out = [f"<b>3 PM trend-hold scan - NSE F&O - {day}</b>\n"
           f"Price has stayed on one side of the EMA20 for the last {window} candles (3 hours), mostly beyond EMA5 and EMA9, "
           f"with EMA50/100/200 in order. Sorted: closest to EMA100 first. Scanned {total} F&O stocks."]
    for side, icon, label in (("BEAR", "🔴", "BEARISH (EMA50 under EMA100 under EMA200, price below EMA5 and EMA9)"),
                              ("BULL", "🟢", "BULLISH (EMA50 over EMA100 over EMA200, price above EMA5 and EMA9)")):
        S = R[R.side == side].copy()
        S["ad"] = S.d100.abs()
        S = S.sort_values("ad")
        out.append(f"\n<b>{icon} {label}: {len(S)}</b>")
        for _, r in S.head(MAX_PER_SIDE).iterrows():
            v = f" vol {r.vol:.1f}x" if r.vol else ""
            out.append(f"{icon} <b>{r.t.replace('.NS', '')}</b> ₹{r.px:,.2f} | {r.d100:+.2f}% from EMA100, {r.d50:+.2f}% from EMA50 | "
                       f"{r.n_strict}/{window} candles beyond EMA5&9 | day {r.day_chg:+.2f}% | {r.vwap_side} VWAP{v}")
        if len(S) > MAX_PER_SIDE:
            out.append(f"... and {len(S) - MAX_PER_SIDE} more further from EMA100")
    out.append("\nNote: a screen to look at, not a proven edge. The 'closer to EMA50 sells more' hypothesis is tracked daily.")
    return out


def log_rows(R, day):
    if R.empty:
        return
    X = R.copy()
    X["source"] = "live"
    X["day"] = day
    if os.path.exists(LOG):
        old = pd.read_csv(LOG)
        old = old[~((old.day == day) & (old.source == "live"))]
        X = pd.concat([old, X], ignore_index=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    X.to_csv(LOG, index=False)


def main():
    dry, force = "--dry" in sys.argv, "--force" in sys.argv
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(TZ))
    day = local.strftime("%Y-%m-%d")
    try:
        sent = json.load(open(SENT))
    except Exception:
        sent = {}
    if sent.get("day") == day and not force:
        print("already sent for", day)
        return
    if local.weekday() >= 5 and not force:
        print("weekend")
        return
    tick = universe()
    data = L.fetch(tick, period="30d")
    R = scan_live(data, now)
    if R.empty:
        L.send(f"<b>3 PM trend-hold scan - NSE F&O - {day}</b>\nNo stock qualifies.", dry)
    else:
        lines = build_message(R, now, len(data))
        chunk, size = [], 0
        for ln in lines:
            if size + len(ln) > 3500 and chunk:
                L.send("\n".join(chunk), dry)
                chunk, size = ["<b>3 PM trend-hold scan (cont.)</b>"], 0
            chunk.append(ln)
            size += len(ln)
        L.send("\n".join(chunk), dry)
    print(f"scanned {len(data)} F&O stocks | bearish {int((R.side == 'BEAR').sum()) if len(R) else 0} | "
          f"bullish {int((R.side == 'BULL').sum()) if len(R) else 0} | latest candle {R.bar.max() if len(R) else '-'}")
    if not dry:
        log_rows(R, R.day.iloc[0] if len(R) else day)
        os.makedirs(os.path.dirname(SENT), exist_ok=True)
        json.dump({"day": day}, open(SENT, "w"))


if __name__ == "__main__":
    L.raise_fd_limit()
    main()
