#!/usr/bin/env python3
"""3 PM trend-hold scan (NSE: F&O stocks + Nifty Smallcap 250 + Hindustan Copper).

BEARISH: for the last WINDOW 15m candles the price has stayed below the EMA20 (the allowed relaxation),
         with at least MIN_SHARE of those candles closing below BOTH EMA5 and EMA9, the latest close is below
         EMA5 and EMA9, and the slow EMAs are descending: EMA50 < EMA100 < EMA200.
BULLISH: the mirror image (above EMA20, mostly above EMA5/EMA9, latest close above both, EMA50 > EMA100 > EMA200).
Lists are sorted with the stock CLOSEST to its EMA100 first (smallest distance, ascending).

Meant to run once a day at ~15:01 IST (launchd), idempotent via learning/trend_scan_sent.json.
Usage: python3 trend_hold_scan.py [--dry] [--force] [--window=12] [--share=0.75]
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import live_candle_alert as L

BASE = os.path.dirname(os.path.abspath(__file__))
SENT = os.path.join(BASE, "learning", "trend_scan_sent.json")
TZ = "Asia/Kolkata"
EXTRA = ["HINDCOPPER.NS"]
MAX_PER_SIDE = 30


def universe():
    tick = list(L.load_markets()["NSE"]["tickers"])
    names = {}
    try:
        sc = pd.read_csv(os.path.join(BASE, "learning", "smallcap250.csv"))
        tick += [s + ".NS" for s in sc.Symbol]
        names = {s + ".NS": n for s, n in zip(sc.Symbol, sc["Company Name"])}
    except Exception as e:
        print("smallcap list unavailable:", e)
    return list(dict.fromkeys(tick + EXTRA)), names


def scan(data, window, share, now):
    rows = []
    for t, df in data.items():
        b = L.completed(df, now).dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if len(b) < 260:
            continue
        b.index = b.index.tz_convert(TZ)
        b["day"] = b.index.strftime("%Y-%m-%d")
        c = b.Close
        e = {p: c.ewm(span=p, adjust=False).mean() for p in (5, 9, 20, 50, 100, 200)}
        last = b.iloc[-1]
        today = b[b.day == last.day]
        w = slice(len(b) - window, len(b))
        cw = c.iloc[w]
        below_both = (cw < np.minimum(e[5].iloc[w], e[9].iloc[w])).sum()
        above_both = (cw > np.maximum(e[5].iloc[w], e[9].iloc[w])).sum()
        e50, e100, e200 = e[50].iloc[-1], e[100].iloc[-1], e[200].iloc[-1]
        px = float(c.iloc[-1])
        side = None
        if (cw < e[20].iloc[w]).all() and below_both >= math.ceil(share * window) and px < min(e[5].iloc[-1], e[9].iloc[-1]) and e50 < e100 < e200:
            side = "BEAR"
        elif (cw > e[20].iloc[w]).all() and above_both >= math.ceil(share * window) and px > max(e[5].iloc[-1], e[9].iloc[-1]) and e50 > e100 > e200:
            side = "BULL"
        if not side:
            continue
        prev = b[b.day < last.day]
        tp = (today.High + today.Low + today.Close) / 3
        vw = (tp * today.Volume).sum() / today.Volume.sum() if today.Volume.sum() > 0 else np.nan
        rows.append(dict(t=t, side=side, px=px, d100=(px / e100 - 1) * 100, d50=(px / e50 - 1) * 100, d20=(px / e[20].iloc[-1] - 1) * 100,
                         n_strict=int(below_both if side == "BEAR" else above_both), day=(px / prev.Close.iloc[-1] - 1) * 100 if len(prev) else np.nan,
                         vwap_side=("below" if px < vw else "above") if vw == vw else "n/a",
                         vol=L.slot_volume_ratio(b, str(b.index[-1])), bar=b.index[-1].strftime("%H:%M")))
    return pd.DataFrame(rows)


def build_message(R, names, window, now, total):
    day = now.astimezone(ZoneInfo_()).strftime("%Y-%m-%d %H:%M")
    head = (f"<b>3 PM trend-hold scan - NSE - {day}</b>\n"
            f"Price has stayed on one side of the EMA20 for the last {window} candles (3 hours), mostly beyond EMA5 and EMA9, "
            f"with EMA50/100/200 in order. Sorted: closest to EMA100 first. Scanned {total} stocks.")
    out = [head]
    for side, icon, label in (("BEAR", "🔴", "BEARISH (EMA50 under EMA100 under EMA200, price below EMA5 and EMA9)"), ("BULL", "🟢", "BULLISH (EMA50 over EMA100 over EMA200, price above EMA5 and EMA9)")):
        S = R[R.side == side].copy()
        S["ad"] = S.d100.abs()
        S = S.sort_values("ad")
        out.append(f"\n<b>{icon} {label}: {len(S)}</b>")
        for _, r in S.head(MAX_PER_SIDE).iterrows():
            sym = r.t.replace(".NS", "")
            nm = f" {names.get(r.t, '')[:20]}" if names.get(r.t) else ""
            v = f" vol {r.vol:.1f}x" if r.vol else ""
            out.append(f"{icon} <b>{sym}</b>{nm} ₹{r.px:,.2f} | {r.d100:+.2f}% from EMA100, {r.d50:+.2f}% from EMA50 | {r.n_strict}/{window} candles beyond EMA5&9 | day {r.day:+.2f}% | {r.vwap_side} VWAP{v}")
        if len(S) > MAX_PER_SIDE:
            out.append(f"... and {len(S) - MAX_PER_SIDE} more further from EMA100")
    out.append("\nNote: a screen to look at, not a proven edge.")
    return out


def ZoneInfo_():
    from zoneinfo import ZoneInfo
    return ZoneInfo(TZ)


def main():
    dry, force = "--dry" in sys.argv, "--force" in sys.argv
    window = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--window=")), 12))
    share = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--share=")), 0.75))
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo_())
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
    tick, names = universe()
    data = L.fetch(tick, period="30d")
    R = scan(data, window, share, now)
    if R.empty:
        L.send(f"<b>3 PM trend-hold scan - NSE - {day}</b>\nNo stock qualifies.", dry)
    else:
        lines = build_message(R, names, window, now, len(data))
        chunk, size = [], 0
        for ln in lines:
            if size + len(ln) > 3500 and chunk:
                L.send("\n".join(chunk), dry)
                chunk, size = ["<b>3 PM trend-hold scan (cont.)</b>"], 0
            chunk.append(ln)
            size += len(ln)
        L.send("\n".join(chunk), dry)
    print(f"scanned {len(data)} | bearish {int((R.side == 'BEAR').sum()) if len(R) else 0} | bullish {int((R.side == 'BULL').sum()) if len(R) else 0} | latest candle {R.bar.max() if len(R) else '-'}")
    if not dry:
        os.makedirs(os.path.dirname(SENT), exist_ok=True)
        json.dump({"day": day}, open(SENT, "w"))


if __name__ == "__main__":
    L.raise_fd_limit()
    main()
