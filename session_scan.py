#!/usr/bin/env python3
"""Opening-range and range-breakout scans for NSE F&O stocks (replaces the old near-extreme / volume-surge candle alerts).

MORNING (sent 09:46, 10:01, 10:16, 10:31, 10:46 - every 15 minutes up to 11:00):
  Opening range = the 09:15-09:30 candle. The candle that just closed (09:30, 09:45, 10:00, 10:15, 10:30) must be a NO-WICK
  candle (body >= 80%, each wick <= 10%) that closes beyond the opening-range high (bull, green) or low (bear, red).
  From the 09:30 candle on (including the 09:46 alert), a no-wick candle that closes within 0.25% short of the range edge is also listed as NEAR.
  Sorted by total wick, smallest first.
AFTERNOON (breakout of the range since 10:00):
  11:16, 12:16, 13:16, 14:16, 15:16 ;  range = 10:00 up to the start of the candle ; candle = the 15m candle just closed
  The candle must close above the range high (bull) / below the range low (bear), body >= 70%, volume >= 2x normal for the slot.
Afternoon messages are sorted by buy/sell pressure in the signal direction (volume of up vs down 5-minute bars inside the
candle; the NSE order book cannot be read by a script).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

IST = "Asia/Kolkata"
MORNING = {st: ("09:15", "09:30") for st in ("09:30", "09:45", "10:00", "10:15", "10:30")}   # candle start -> (range start, range end): always the opening 15 minutes
NEAR_FROM = "09:30"      # near-breakout counts from the first candle on (09:46 alert included)
NEAR_PCT = 0.25          # "near" = closes within this % of the opening-range edge
WICK_MAX = 10.0          # each wick <= 10% of the candle range
BODY_AM = 80.0
AFTERNOON = ("11:00", "12:00", "13:00", "14:00", "15:00")
RANGE_START_PM = "10:00"
ZONE = 0.25
BODY_PM = 70.0
VOL_PM = 2.0
MIN_RANGE_PCT = 0.15
MAX_AGE_MIN = 35
TOP_N = 15          # per message; the rest are counted, not listed


def _local(df):
    d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    d.index = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(IST)
    d["day"] = d.index.strftime("%Y-%m-%d")
    d["hm"] = d.index.strftime("%H:%M")
    return d


def pressure(sym_ns, bar_start):
    """Signed volume share (-100..+100) of the three 5-minute bars inside the 15m candle starting at bar_start (IST Timestamp)."""
    try:
        d = yf.download(sym_ns, period="1d", interval="5m", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d.index = d.index.tz_convert(IST)
        w = d[(d.index >= bar_start) & (d.index < bar_start + timedelta(minutes=15))]
        v = w.Volume.astype(float)
        sgn = np.sign(w.Close - w.Open)
        tot = v.sum()
        return float((v * sgn).sum() / tot * 100) if tot > 0 else None
    except Exception:
        return None


def scan(H, data, now, state, dry):
    """H = the live_candle_alert module (helpers). data = {ticker: 15m df}. Returns number of alerts sent."""
    nowi = now.astimezone(ZoneInfo(IST))
    cands, keys = [], []
    for t, df in data.items():
        d = _local(df)
        if d.empty:
            continue
        day = d.day.iloc[-1]
        g = d[d.day == day]
        for start in list(MORNING) + list(AFTERNOON):
            c = g[g.hm == start]
            if c.empty:
                continue
            c = c.iloc[0]
            close_at = c.name + timedelta(minutes=15)
            age = (nowi - close_at).total_seconds() / 60
            if age < 0 or age > MAX_AGE_MIN:
                continue
            if start in MORNING:
                r0, r1 = MORNING[start]
                rg = g[(g.hm >= r0) & (g.hm < r1)]
                body_min, am = BODY_AM, True
            else:
                rg = g[(g.hm >= RANGE_START_PM) & (g.hm < start)]
                body_min, am = BODY_PM, False
            if rg.empty:
                continue
            hi, lo = float(rg.High.max()), float(rg.Low.min())
            if hi <= lo or (hi / lo - 1) * 100 < MIN_RANGE_PCT:
                continue
            o, h, l, cl = float(c.Open), float(c.High), float(c.Low), float(c.Close)
            body = abs(cl - o) / max(h - l, 1e-9) * 100
            if body < body_min:
                continue
            upw = (h - max(o, cl)) / max(h - l, 1e-9) * 100
            low_w = (min(o, cl) - l) / max(h - l, 1e-9) * 100
            if am and (upw > WICK_MAX or low_w > WICK_MAX):
                continue
            pos = (cl - lo) / (hi - lo)
            near = False
            if am:
                nz = NEAR_PCT / 100 if start >= NEAR_FROM else 0.0
                bull = cl > o and cl >= hi * (1 - nz)
                bear = cl < o and cl <= lo * (1 + nz)
                near = (bull and cl <= hi) or (bear and cl >= lo)
            else:
                bull, bear = cl > o and cl > hi, cl < o and cl < lo
            if not (bull or bear):
                continue
            dr = "bull" if bull else "bear"
            key = f"RNG|{t}|{day}|{start}|{dr}"
            if key in state:
                continue
            cands.append(dict(t=t, start=start, dir=dr, am=am, near=near, wick=upw + low_w, upw=upw, low_w=low_w, hi=hi, lo=lo, pos=pos, body=body, px=cl, bar=c.name,
                              day=day, prev=float(d[d.day < day].Close.iloc[-1]) if (d.day < day).any() else None))
            keys.append(key)
    if not cands:
        return 0
    long_data = H.fetch(sorted({p["t"] for p in cands}), period="30d")
    keep = []
    for p in cands:
        ldf = long_data.get(p["t"])
        p["vx"] = H.slot_volume_ratio(ldf, p["bar"]) if ldf is not None else None
        if not p["am"] and (p["vx"] is None or p["vx"] < VOL_PM):
            continue
        p["tilt"] = pressure(p["t"], p["bar"])
        s = p["tilt"] if p["tilt"] is not None else 0.0
        p["score"] = s if p["dir"] == "bull" else -s
        keep.append(p)
    if not keep:
        for k in keys:
            state[k] = now.isoformat()      # evaluated and rejected (volume): do not re-check every cycle
        return 0
    oi_ctx = H.oi_context(nowi)
    sent = []
    by = {}
    for p in keep:
        by.setdefault(p["start"], []).append(p)
    messages = []
    for start, grp in sorted(by.items()):
        grp.sort(key=(lambda p: (p["wick"], -p["score"])) if start in MORNING else (lambda p: (-p["score"], -(p["vx"] or 0) * p["body"])))
        total, grp = len(grp), grp[:TOP_N]
        if start in MORNING:
            r0, r1 = MORNING[start]
            title = f"opening range {r0}-{r1}, no-wick candle {start}-{(datetime.strptime(start, '%H:%M') + timedelta(minutes=15)).strftime('%H:%M')} (least wick first)"
        else:
            title = f"10:00-{start} range breakout, candle {start}"
        srt = "Sorted: smallest wicks first." if start in MORNING else "Sorted: strongest buy/sell pressure first (volume of up vs down 5-minute bars; the NSE order book is not readable by scripts)."
        head = f"<b>📐 Range scan - {title}</b>\n{srt}\n"
        lines = []
        for p in grp:
            sym = p["t"].replace(".NS", "")
            icon = "🟢" if p["dir"] == "bull" else "🔴"
            what = "LONG" if p["dir"] == "bull" else "SHORT"
            if p["am"]:
                edge = p["hi"] if p["dir"] == "bull" else p["lo"]
                gap = abs(p["px"] / edge - 1) * 100
                where = (f"NEAR opening-range {'high' if p['dir'] == 'bull' else 'low'} {edge:,.2f} ({gap:.2f}% short of it)" if p["near"] else
                         f"BROKE opening-range {'high' if p['dir'] == 'bull' else 'low'} {edge:,.2f} (by {gap:.2f}%)") + f", wicks up {p['upw']:.0f}% / down {p['low_w']:.0f}%"
            else:
                where = ("above" if p["dir"] == "bull" else "below") + f" the range {p['lo']:,.2f}-{p['hi']:,.2f}"
            tilt = f"{p['tilt']:+.0f}% {'buy' if p['tilt'] > 0 else 'sell'} pressure" if p["tilt"] is not None else "pressure n/a"
            vol = f" | volume {p['vx']:.1f}x normal" if p["vx"] else ""
            dchg = (p["px"] / p["prev"] - 1) * 100 if p["prev"] else 0.0
            oi = H.oi_note(sym, p["dir"], dchg, oi_ctx)
            lines.append(f"{icon} <b>{sym}</b> Rs {p['px']:,.2f} {what} setup - {where}, body {p['body']:.0f}% | {tilt}{vol}{oi}")
            sent.append({"ts": now.isoformat(), "market": "NSE", "date": p["day"], "bar": str(p["bar"]), "t": p["t"], "sym": sym,
                         "kind": "RANGE", "dir": p["dir"], "px": p["px"]})
        if total > len(grp):
            lines.append(f"... and {total - len(grp)} more passed the rule - not listed")
        messages.append((head, lines))
    try:
        for head, lines in messages:
            H.send_chunks(head, lines, dry)
    except Exception as e:
        print("RANGE SEND FAILED:", e)
        return 0
    for k in keys:
        state[k] = now.isoformat()
    if not dry:
        H.append_log(sent)
    return len(sent)
