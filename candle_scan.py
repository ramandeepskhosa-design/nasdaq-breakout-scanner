#!/usr/bin/env python3
"""The two alerts that are left after the 28 Oct clean-up (everything else is muted via learning/ALERTS_OFF).

PART 1 - 1-minute no-wick scan (python3 candle_scan.py minute), run by launchd at 09:16, 09:31, 09:46, 10:01 IST.
  Looks at the 1-minute candle that opened at 09:15 / 09:30 / 09:45 / 10:00 (just closed) for every NSE F&O stock:
    bullish = GREEN candle with no lower wick (open = low)   | bearish = RED candle with no upper wick (open = high)
    plus body >= 60% of the candle range and move >= 0.2% (otherwise hundreds of stocks qualify on tiny candles).
  Sent about a minute after the candle closes (09:17, 09:32, 09:47, 10:02), strongest turnover first.

PART 2 - 15-minute EMA-pullback candles (called from live_candle_alert.run_cycle at :01/:16/:31/:46), candles 09:15 to 15:00:
    bull: EMA20>50>100>200, green candle, lower wick <= 7%, low touches/near EMA5 or EMA9, volume >= 1.2x normal
    bear: EMA20<50<100<200, red candle, upper wick <= 7%, high touches/near EMA5 or EMA9, volume >= 1.2x normal
  both: body >= 40% and the far-side wick <= 30%
  First alert ~09:31 (the 09:15 candle), then every 15 minutes.
"""
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"
MIN_BODY_1M, MIN_MOVE_1M, WICK_TOL = 60.0, 0.20, 2.0     # 1-minute scan: body %, move %, "no wick" = wick <= 2% of the range
VOL_MIN, MAX_WICK, NEAR_PCT = 1.2, 7.0, 0.15     # 15m scan: volume x normal, wick % of range, "near EMA" = within 0.15% of price
MIN_BODY, MAX_OPP_WICK = 40.0, 30.0     # 15m scan: body >= 40% of range, wick on the far side <= 30%
FIRST_15M = "09:15"
LAST_15M = "15:00"
MIN_RANGE_15M = 0.20     # % - ignore tiny candles
TOP_N = 20
MINUTE_STARTS = ("09:15", "09:30", "09:45", "10:00")


def _load(H, ticks):
    return ticks


def minute_scan(dry=False):
    import live_candle_alert as H
    now = datetime.now(ZoneInfo(IST))
    # launchd can start late (sleep/wake): pick the latest scan slot whose 1-minute candle closed within the last 12 minutes
    want = None
    for st in MINUTE_STARTS:
        closes = datetime.strptime(now.strftime("%Y-%m-%d ") + st, "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo) + timedelta(minutes=1)
        if timedelta(0) <= now - closes <= timedelta(minutes=12):
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
    # candle opened at `want`; Yahoo publishes it 20-60 s late. Read Yahoo's chart API directly (yfinance's sqlite cache breaks under threads:
    # "unable to open database file" silently dropped a third of the stocks), retrying the tickers still missing.
    import requests
    from concurrent.futures import ThreadPoolExecutor
    rng = "1d" if day == now.strftime("%Y-%m-%d") else "5d"

    def one(t):
        for _ in range(2):
            try:
                r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t}", params={"interval": "1m", "range": rng},
                                 headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if r.status_code != 200:
                    time.sleep(1.5)
                    continue
                j = r.json()["chart"]["result"][0]
                q = j["indicators"]["quote"][0]
                df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"], "Close": q["close"], "Volume": q["volume"]},
                                  index=pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(IST)).dropna(subset=["Open", "High", "Low", "Close"])
                c = df[(df.index.strftime("%Y-%m-%d") == day) & (df.index.strftime("%H:%M") == want)]
                return t, (c.iloc[0] if len(c) else None)
            except Exception:
                time.sleep(1)
        return t, None

    rows = {}
    deadline = time.time() + 150
    while time.time() < deadline:
        missing = [t for t in ticks if t not in rows]
        with ThreadPoolExecutor(12) as ex:
            for t, c in ex.map(one, missing):
                if c is not None:
                    rows[t] = (c, None)
        if len(rows) >= 0.93 * len(ticks) or day != now.strftime("%Y-%m-%d"):
            break
        time.sleep(6)
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
    """Part 2 (rewritten 30 Oct). 15-minute EMA-pullback candles, every 15 minutes from the 09:15 candle (alert ~09:31) to 15:00.
    Bull: EMA20 > EMA50 > EMA100 > EMA200, GREEN candle, lower wick <= 7% of its range, low touched / within NEAR_PCT of EMA5 or EMA9, volume >= 1.2x normal.
    Bear: EMA20 < EMA50 < EMA100 < EMA200, RED candle, upper wick <= 7%, high touched / within NEAR_PCT of EMA5 or EMA9, volume >= 1.2x normal.
    `data` only supplies the ticker list; EMAs need ~30 days of 15m bars, so they are fetched here. Returns number of alerts sent."""
    nowi = now.astimezone(ZoneInfo(IST))
    long = H.fetch(sorted(data), period="30d")
    found = []
    for t, df in long.items():
        d = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if len(d) < 220:
            continue
        d.index = d.index.tz_convert(IST) if d.index.tz is not None else d.index.tz_localize("UTC").tz_convert(IST)
        ema = {n: d.Close.ewm(span=n, adjust=False).mean() for n in (5, 9, 20, 50, 100, 200)}
        day = d.index[-1].strftime("%Y-%m-%d")
        for k in range(len(d)):
            if d.index[k].strftime("%Y-%m-%d") != day:
                continue
            st = d.index[k].strftime("%H:%M")
            if st < FIRST_15M or st > LAST_15M:
                continue
            age = (nowi - (d.index[k] + timedelta(minutes=15))).total_seconds() / 60
            if age < 0 or age > 35:
                continue
            c = d.iloc[k]
            o, h, l, cl = float(c.Open), float(c.High), float(c.Low), float(c.Close)
            r = h - l
            if r <= 0 or (r / o) * 100 < MIN_RANGE_15M or cl == o:
                continue
            e5, e9, e20, e50, e100, e200 = (float(ema[n].iloc[k]) for n in (5, 9, 20, 50, 100, 200))
            if cl > o:
                col = "bull"
                if not (e20 > e50 > e100 > e200):
                    continue
                wick = (min(o, cl) - l) / r * 100
                opp = (h - max(o, cl)) / r * 100
                dist = min(abs(l - e5) / e5, abs(l - e9) / e9) * 100 if not (l <= e5 <= h or l <= e9 <= h) else 0.0
            else:
                col = "bear"
                if not (e20 < e50 < e100 < e200):
                    continue
                wick = (h - max(o, cl)) / r * 100
                opp = (min(o, cl) - l) / r * 100
                dist = min(abs(h - e5) / e5, abs(h - e9) / e9) * 100 if not (l <= e5 <= h or l <= e9 <= h) else 0.0
            if wick > MAX_WICK or opp > MAX_OPP_WICK or abs(cl - o) / r * 100 < MIN_BODY or dist > NEAR_PCT:
                continue
            vx = H.slot_volume_ratio(d, d.index[k])
            if vx is None or vx < VOL_MIN:
                continue
            key = f"C15|{t}|{day}|{st}|{col}"
            if key in state:
                continue
            rule = f"EMA {'20 above 50 above 100 above 200' if col == 'bull' else '20 below 50 below 100 below 200'}, body {abs(cl - o) / r * 100:.0f}%, {'lower' if col == 'bull' else 'upper'} wick {wick:.0f}%, {'touched' if dist == 0 else f'{dist:.2f}% from'} EMA5/9, volume {vx:.1f}x"
            prevd = d[d.index.strftime("%Y-%m-%d") < day]
            found.append(dict(t=t, key=key, dir=col, rule=rule, px=cl, body=abs(cl - o) / r * 100, vx=vx, start=st, prev=float(prevd.Close.iloc[-1]) if len(prevd) else None, bar=d.index[k]))
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
            grp.sort(key=lambda x: -(x["vx"] or 0))
            tot, grp = len(grp), grp[:TOP_N]
            end = (datetime.strptime(st, "%H:%M") + timedelta(minutes=15)).strftime("%H:%M")
            icon = "🟢" if d == "bull" else "🔴"
            head = f"<b>{icon} 15-minute {'GREEN, EMAs rising' if d == 'bull' else 'RED, EMAs falling'}: pullback to EMA5/9, candle {st}-{end}</b>\nTrend-aligned candle with a tiny {'lower' if d == 'bull' else 'upper'} wick and 1.2x volume. Screen only.\n"
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
