#!/usr/bin/env python3
"""The two alerts that are left after the 28 Oct clean-up (everything else is muted via learning/ALERTS_OFF).

PART 1 - 1-minute no-wick scan (python3 candle_scan.py minute), run by launchd at 09:16, 09:31, 09:46, 10:01 IST.
  Looks at the 1-minute candle that opened at 09:15 / 09:30 / 09:45 / 10:00 (just closed) for every NSE F&O stock:
    bullish = GREEN candle with no lower wick (open = low)   | bearish = RED candle with no upper wick (open = high)
    plus body >= 60% of the candle range and move >= 0.2% (otherwise hundreds of stocks qualify on tiny candles).
  Sent about a minute after the candle closes (09:17, 09:32, 09:47, 10:02), strongest turnover first.

PART 2 - 15-minute trend candles (called from live_candle_alert.run_cycle at :01/:16/:31/:46), candles from 10:30 on:
    A) one candle with body >= 70% and volume >= 1.5x normal for that time, or
    B) two candles in a row of the same colour, each with body >= 60%,
  bull (green) and bear (red). First alert 10:46 (10:30 candle), last 15:16 (15:00 candle... 15:15 candle is thin, skipped).
"""
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"
MIN_BODY_1M, MIN_MOVE_1M, WICK_TOL = 60.0, 0.20, 2.0     # 1-minute scan: body %, move %, "no wick" = wick <= 2% of the range
BODY_A, VOL_A, BODY_B = 70.0, 1.5, 60.0
FIRST_15M = "10:30"
LAST_15M = "15:00"
MIN_RANGE_15M = 0.25     # % - ignore tiny candles
TOP_N = 20
MINUTE_STARTS = ("09:15", "09:30", "09:45", "10:00")


def _load(H, ticks):
    return ticks


def minute_scan(dry=False):
    import live_candle_alert as H
    now = datetime.now(ZoneInfo(IST))
    # launchd can start late (sleep/wake): pick the latest scan slot whose 1-minute candle closed within the last 6 minutes
    want = None
    for st in MINUTE_STARTS:
        closes = datetime.strptime(now.strftime("%Y-%m-%d ") + st, "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo) + timedelta(minutes=1)
        if timedelta(0) <= now - closes <= timedelta(minutes=6):
            want = st
    if len(sys.argv) > 3:
        want = sys.argv[3]
    if want not in MINUTE_STARTS:
        print("not a scan minute:", now.strftime("%H:%M"))
        return
    day = now.strftime("%Y-%m-%d")
    if len(sys.argv) > 2 and sys.argv[2] != "now":
        day = sys.argv[2]
    ticks = list(H.load_markets()["NSE"]["tickers"])
    # candle opened at `want`; Yahoo publishes it 20-60 s late and rate-limits bursts, so keep retrying the tickers still missing
    import os
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learning", "yf_cache")
    os.makedirs(cache, exist_ok=True)
    try:
        yf.set_tz_cache_location(cache)
    except Exception:
        pass
    rows = {}
    deadline = time.time() + 150
    while time.time() < deadline:
        missing = [t for t in ticks if t not in rows]
        for i in range(0, len(missing), 40):
            part = missing[i:i + 40]
            try:
                d = yf.download(part, period=("1d" if day == now.strftime("%Y-%m-%d") else "5d"), interval="1m", group_by="ticker", threads=False, progress=False, auto_adjust=False)
            except Exception:
                time.sleep(3)
                continue
            for t in part:
                try:
                    df = d[t].dropna(subset=["Open", "High", "Low", "Close"])
                    df.index = df.index.tz_convert(IST)
                    c = df[(df.index.strftime("%Y-%m-%d") == day) & (df.index.strftime("%H:%M") == want)]
                    if len(c):
                        rows[t] = (c.iloc[0], df)
                except Exception:
                    pass
        if len(rows) >= 0.9 * len(ticks) or day != now.strftime("%Y-%m-%d"):
            break
        time.sleep(8)
    out = []
    for t, (c, df) in rows.items():
        o, h, l, cl = float(c.Open), float(c.High), float(c.Low), float(c.Close)
        r = h - l
        if r <= 0 or o <= 0:
            continue
        body = abs(cl - o) / r * 100
        move = (cl / o - 1) * 100
        lw = (min(o, cl) - l) / r * 100
        uw = (h - max(o, cl)) / r * 100
        if body < MIN_BODY_1M or abs(move) < MIN_MOVE_1M:
            continue
        if cl > o and lw <= WICK_TOL:
            d = "bull"
        elif cl < o and uw <= WICK_TOL:
            d = "bear"
        else:
            continue
        out.append(dict(sym=t.replace(".NS", ""), dir=d, o=o, cl=cl, move=move, body=body, turn=float(c.Volume) * cl, vol=int(c.Volume)))
    out.sort(key=lambda x: (-x["turn"], -abs(x["move"])))   # 09:15 minute has no volume on Yahoo: then biggest move first
    print(f"{want}: {len(rows)} stocks read, {len(out)} no-wick 1-minute candles")
    if not out:
        return
    H._ALLOW = True
    end = (datetime.strptime(want, "%H:%M") + timedelta(minutes=1)).strftime("%H:%M")
    for d, icon, title in (("bull", "🟢", "BULLISH - green, no lower wick"), ("bear", "🔴", "BEARISH - red, no upper wick")):
        g = [x for x in out if x["dir"] == d][:TOP_N]
        tot = len([x for x in out if x["dir"] == d])
        if not g:
            continue
        lines = [f"{icon} <b>{x['sym']}</b> Rs {x['cl']:,.2f}  {x['move']:+.2f}% in 1 min, body {x['body']:.0f}%" + (f", turnover Rs {x['turn'] / 1e7:.2f} cr" if x["turn"] > 0 else "") for x in g]
        if tot > len(g):
            lines.append(f"... and {tot - len(g)} more (smaller turnover)")
        head = f"<b>1-minute no-wick candles {want}-{end} - {title}</b>\nBiggest turnover first (biggest move first at 09:15, when volume is not yet published). Screen only.\n"
        try:
            H.send_chunks(head, lines, dry)
        finally:
            H._ALLOW = False


def scan15(H, data, now, state, dry):
    """Part 2. data = {ticker: 15m df (2d)}. Returns number of alerts sent."""
    nowi = now.astimezone(ZoneInfo(IST))
    found = []
    for t, df in data.items():
        d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if d.empty:
            continue
        d.index = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(IST)
        day = d.index[-1].strftime("%Y-%m-%d")
        g = d[d.index.strftime("%Y-%m-%d") == day]
        hms = g.index.strftime("%H:%M")
        for k in range(len(g)):
            c = g.iloc[k]
            st = hms[k]
            if st < FIRST_15M or st > LAST_15M:
                continue
            age = (nowi - (g.index[k] + timedelta(minutes=15))).total_seconds() / 60
            if age < 0 or age > 35:
                continue
            o, h, l, cl = float(c.Open), float(c.High), float(c.Low), float(c.Close)
            r = h - l
            if r <= 0 or (r / o) * 100 < MIN_RANGE_15M:
                continue
            body = abs(cl - o) / r * 100
            col = "bull" if cl > o else "bear" if cl < o else None
            if col is None:
                continue
            rule = None
            vx = None
            if body >= BODY_A:
                vx = H.slot_volume_ratio(df, g.index[k])
                if vx is not None and vx >= VOL_A:
                    rule = f"A: body {body:.0f}% with volume {vx:.1f}x normal"
            if rule is None and body >= BODY_B and k >= 1:
                p = g.iloc[k - 1]
                pr = float(p.High) - float(p.Low)
                if pr > 0 and abs(float(p.Close) - float(p.Open)) / pr * 100 >= BODY_B and (float(p.Close) > float(p.Open)) == (cl > o):
                    rule = f"B: 2 {'green' if col == 'bull' else 'red'} candles in a row, bodies {abs(float(p.Close) - float(p.Open)) / pr * 100:.0f}% + {body:.0f}%"
            if rule is None:
                continue
            key = f"C15|{t}|{day}|{st}|{col}"
            if key in state:
                continue
            found.append(dict(t=t, key=key, dir=col, rule=rule, px=cl, body=body, vx=vx, start=st, prev=float(d[d.index.strftime("%Y-%m-%d") < day].Close.iloc[-1]) if (d.index.strftime("%Y-%m-%d") < day).any() else None, bar=g.index[k]))
    if not found:
        return 0
    oi_ctx = H.oi_context(nowi)
    by = {}
    for f in found:
        by.setdefault((f["start"], f["dir"]), []).append(f)
    H._ALLOW = True
    sent = []
    try:
        for (st, d), grp in sorted(by.items()):
            grp.sort(key=lambda x: (x["rule"][0] != "A", -(x["vx"] or 0), -x["body"]))
            tot, grp = len(grp), grp[:TOP_N]
            end = (datetime.strptime(st, "%H:%M") + timedelta(minutes=15)).strftime("%H:%M")
            icon = "🟢" if d == "bull" else "🔴"
            head = f"<b>{icon} 15-minute {'GREEN' if d == 'bull' else 'RED'} trend candles {st}-{end}</b>\nA = 70% body + 1.5x volume, B = two same-colour candles with 60% bodies. Screen only.\n"
            lines = []
            for f in grp:
                sym = f["t"].replace(".NS", "")
                dchg = (f["px"] / f["prev"] - 1) * 100 if f["prev"] else 0.0
                lines.append(f"{icon} <b>{sym}</b> Rs {f['px']:,.2f} (day {dchg:+.1f}%) - {f['rule']}{H.oi_note(sym, d, dchg, oi_ctx)}")
                sent.append({"ts": now.isoformat(), "market": "NSE", "date": f["bar"].strftime("%Y-%m-%d"), "bar": str(f["bar"]), "t": f["t"], "sym": sym, "kind": "TREND15", "dir": d, "px": f["px"]})
            if tot > len(grp):
                lines.append(f"... and {tot - len(grp)} more")
            H.send_chunks(head, lines, dry)
    except Exception as e:
        print("CANDLE15 SEND FAILED:", e)
        return 0
    finally:
        H._ALLOW = False
    for f in found:
        state[f["key"]] = now.isoformat()
    if not dry:
        H.append_log(sent)
    return len(sent)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "minute":
        minute_scan("--dry" in sys.argv)
