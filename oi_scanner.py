#!/usr/bin/env python3
"""15-minute futures-OI and option scanner for NSE F&O stocks (the OI counterpart of the 15-minute no-wick candle alert).

Every 15 minutes (launchd at :02 :17 :32 :47, 09:17-15:32 IST, weekdays):
  1. ONE NSE request gives futures open interest + price for every F&O stock. It is compared with the snapshot ~15 minutes
     earlier (learning/oi_log.csv, written by this scanner and by oi_snapshot.py).
  2. TYPE 1 - OI shift: OI changed by >= OI_PCT percent more than the market median in 15 minutes (min MIN_OI contracts) AND the
     price moved >= MIN_PX percent: new shorts (OI up, price down), new longs (OI up, price up), short covering (OI down, price
     up), long unwinding (OI down, price down).
  3. TYPE 2 - option strike surge: for triggered stocks (and a small watchlist) the option chain is read; a strike whose OI grew
     >= STRIKE_PCT percent and >= STRIKE_MIN contracts since the previous 15-minute snapshot (falls back to today's change vs
     yesterday's close when there is no earlier snapshot), plus premium change and the put/call-ratio shift.
  4. TYPE 3 - confluence: a candle alert (no-wick / near day low-high / volume surge) for the same stock in the last 25 minutes in
     the same direction.
Everything is logged (learning/oi_signals.csv, learning/chain15_log.csv) for the forward test:  python3 oi_scanner.py track

Usage: python3 oi_scanner.py [scan] [--dry] [--force]      python3 oi_scanner.py track [--dry]
Telegram messages are plain text on purpose (no angle brackets)."""
import csv
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import oi_snapshot as S

BASE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(BASE, "learning")
OI_LOG = os.path.join(LEARN, "oi_log.csv")
CHAIN_LOG = os.path.join(LEARN, "chain15_log.csv")
SIGNALS = os.path.join(LEARN, "oi_signals.csv")
STATE = os.path.join(LEARN, "oi_scan_state.json")
ALERT_LOG = os.path.join(BASE, ".live_alert_log.jsonl")
TZ = ZoneInfo("Asia/Kolkata")
INDEXES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
OI_PCT, MIN_OI, MIN_PX, COOLDOWN_MIN = 2.0, 5000, 0.3, 30
STRIKE_PCT, STRIKE_MIN, PCR_SHIFT, MAX_CHAINS, MAX_WATCH = 30.0, 150, 0.10, 12, 6
CLASSES = {"NEW SHORTS": ("short build-up", "bear"), "NEW LONGS": ("long build-up", "bull"),
           "SHORT COVERING": ("short covering", "bull"), "LONG UNWINDING": ("long unwinding", "bear")}


def classify(oi_pct, px_pct):
    if oi_pct > 0:
        return "NEW SHORTS" if px_pct < 0 else "NEW LONGS"
    return "SHORT COVERING" if px_pct > 0 else "LONG UNWINDING"


def read_state():
    try:
        return json.load(open(STATE))
    except Exception:
        return {}


def append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def get_spurts(sess):
    j = sess.get("https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings", timeout=20).json()
    df = pd.DataFrame(j.get("data", []))
    return df[["symbol", "latestOI", "prevOI", "changeInOI", "avgInOI", "volume", "underlyingValue"]]


def prev_snapshot(now, lo=9, hi=28):
    """The OI snapshot closest to 15 minutes ago (between lo and hi minutes ago) from today's log."""
    if not os.path.exists(OI_LOG):
        return None, None
    o = pd.read_csv(OI_LOG, usecols=["ts", "symbol", "latestOI", "underlying"])
    o["t"] = pd.to_datetime(o.ts)
    n = pd.Timestamp(now.replace(tzinfo=None))
    o = o[(o.t >= n - pd.Timedelta(minutes=hi)) & (o.t <= n - pd.Timedelta(minutes=lo))]
    if o.empty:
        return None, None
    target = n - pd.Timedelta(minutes=15)
    best = o.iloc[(o.t - target).abs().argsort()[:1]].t.iloc[0]
    s = o[o.t == best].drop_duplicates("symbol").set_index("symbol")
    return s, best


def log_oi(now, df):
    ts = now.strftime("%Y-%m-%d %H:%M")
    try:
        last = open(OI_LOG).readlines()[-1].split(",")[0] if os.path.exists(OI_LOG) else ""
        if last and (pd.Timestamp(ts) - pd.Timestamp(last)).total_seconds() < 420:
            return
    except Exception:
        pass
    append(OI_LOG, ["ts", "symbol", "latestOI", "prevOI", "changeInOI", "avgInOIpct", "volume", "underlying"],
           [[ts, r.symbol, r.latestOI, r.prevOI, r.changeInOI, r.avgInOI, r.volume, r.underlyingValue] for r in df.itertuples()])


def fetch_chain(sess, sym):
    info = sess.get(f"https://www.nseindia.com/api/option-chain-contract-info?symbol={sym}", timeout=15).json()
    exp = info["expiryDates"][0]
    rec = sess.get(f"https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={sym}&expiry={exp}", timeout=20).json()["records"]
    rows = []
    for r in rec["data"]:
        ce, pe = r.get("CE", {}), r.get("PE", {})
        rows.append(dict(k=r["strikePrice"], ce_oi=ce.get("openInterest", 0), ce_chg=ce.get("changeinOpenInterest", 0), ce_vol=ce.get("totalTradedVolume", 0),
                         ce_ltp=ce.get("lastPrice", 0), ce_iv=ce.get("impliedVolatility", 0),
                         pe_oi=pe.get("openInterest", 0), pe_chg=pe.get("changeinOpenInterest", 0), pe_vol=pe.get("totalTradedVolume", 0),
                         pe_ltp=pe.get("lastPrice", 0), pe_iv=pe.get("impliedVolatility", 0)))
    return exp, float(rec["underlyingValue"]), pd.DataFrame(rows)


def prev_chain(now, sym, exp, lo=9, hi=28):
    if not os.path.exists(CHAIN_LOG):
        return None
    c = pd.read_csv(CHAIN_LOG)
    c = c[(c.symbol == sym) & (c.expiry == exp)]
    if c.empty:
        return None
    c["t"] = pd.to_datetime(c.ts)
    n = pd.Timestamp(now.replace(tzinfo=None))
    c = c[(c.t >= n - pd.Timedelta(minutes=hi)) & (c.t <= n - pd.Timedelta(minutes=lo))]
    if c.empty:
        return None
    best = c.iloc[(c.t - (n - pd.Timedelta(minutes=15))).abs().argsort()[:1]].t.iloc[0]
    return c[c.t == best].set_index("strike")


def option_lines(now, sym, exp, spot, C):
    """Strike surges since the previous 15-minute snapshot (or vs yesterday's close if none) + PCR shift."""
    prev = prev_chain(now, sym, exp)
    out, tag = [], "15m"
    parts = []
    for side in ("ce", "pe"):
        for _, r in C.iterrows():
            cur = r[f"{side}_oi"]
            if prev is not None and r.k in prev.index:
                p0 = float(prev.loc[r.k, f"{side}_oi"]); d = cur - p0; pct = d / p0 * 100 if p0 > 0 else (999 if d > 0 else 0)
                l0 = float(prev.loc[r.k, f"{side}_ltp"]); prem = (r[f"{side}_ltp"] / l0 - 1) * 100 if l0 > 0 else float("nan")
            else:
                tag = "today"
                d = r[f"{side}_chg"]; p0 = cur - d; pct = d / p0 * 100 if p0 > 0 else (999 if d > 0 else 0); prem = float("nan")
            if d >= STRIKE_MIN and pct >= STRIKE_PCT and r[f"{side}_vol"] >= 500:
                pct_txt = "new strike activity" if p0 < STRIKE_MIN * 0.2 else f"{pct:+.0f}%"
                parts.append((d, f"{r.k:g} {side.upper()} {d:+,.0f} ({pct_txt}){'' if prem != prem else f' premium {prem:+.0f}%'} vol {r[f'{side}_vol']:,.0f}", side))
    parts.sort(key=lambda x: -x[0])
    pcr = C.pe_oi.sum() / max(C.ce_oi.sum(), 1)
    pcr_txt = f"PCR {pcr:.2f}"
    if prev is not None and len(prev):
        p_pcr = prev.pe_oi.sum() / max(prev.ce_oi.sum(), 1)
        pcr_txt += f" ({pcr - p_pcr:+.2f} in 15m{' - SHIFT' if abs(pcr - p_pcr) >= PCR_SHIFT else ''})"
    surge = "; ".join(p[1] for p in parts[:3])
    calls = sum(1 for p in parts if p[2] == "ce")
    puts = sum(1 for p in parts if p[2] == "pe")
    return (f"   options ({tag} change): {surge if surge else 'no strike surge'} | {pcr_txt}", calls, puts)


def recent_candle_alerts(now):
    out = {}
    try:
        lines = open(ALERT_LOG).readlines()[-400:]
        n = now.astimezone(ZoneInfo("UTC"))
        for l in lines:
            a = json.loads(l)
            t = datetime.fromisoformat(a["ts"])
            if (n - t).total_seconds() <= 25 * 60 and a.get("market") == "NSE":
                out.setdefault(a["sym"], []).append((a["kind"], a["dir"], a["bar"][11:16]))
    except Exception:
        pass
    return out


def scan(dry=False, force=False):
    now = datetime.now(TZ)
    hm = now.strftime("%H:%M")
    if not force and (now.weekday() >= 5 or not ("09:17" <= hm <= "15:32")):
        return
    slot = now.strftime("%Y-%m-%d ") + f"{now.hour:02d}:{now.minute // 15 * 15:02d}"
    st = read_state()
    if st.get("slot") == slot and not force:
        print("already scanned", slot)
        return
    for attempt in range(3):
        try:
            sess = S.session()
            break
        except Exception as e:
            if attempt == 2:
                raise
            print("NSE session failed, retrying:", str(e)[:80])
            time.sleep(20)
    cur = get_spurts(sess)
    prev, prev_t = prev_snapshot(now.replace(tzinfo=None) if False else now)
    if not dry:
        log_oi(now, cur)
    if prev is None:
        print("no snapshot ~15 minutes ago yet - logged, nothing to compare")
        json.dump({"slot": slot}, open(STATE, "w")) if not dry else None
        return
    mins = int((pd.Timestamp(now.replace(tzinfo=None)) - prev_t).total_seconds() / 60)
    D = cur.set_index("symbol").join(prev.rename(columns={"latestOI": "oi0", "underlying": "px0"}), how="inner")
    D = D[D.oi0 > 0]
    D["pct"] = (D.latestOI - D.oi0) / D.oi0 * 100
    D["px"] = (D.underlyingValue / D.px0 - 1) * 100
    stocks = D[(~D.index.isin(INDEXES)) & (D.latestOI >= MIN_OI)].copy()
    med_oi, med_px = float(stocks.pct.median()), float(stocks.px.median())
    stocks["adj"] = stocks.pct - med_oi
    stocks["rel"] = stocks.px - med_px
    trig = stocks[(stocks.adj.abs() >= OI_PCT) & (stocks.px.abs() >= MIN_PX)].copy()
    done = st.get("cool", {})
    day = now.strftime("%Y-%m-%d")
    ok = pd.Series([(day + "|" + s) not in done or (time.time() - done[day + "|" + s]) > COOLDOWN_MIN * 60 for s in trig.index], index=trig.index, dtype=bool)
    trig = trig[ok]
    trig["abs"] = trig.adj.abs()
    trig = trig.sort_values("abs", ascending=False)
    idx_txt = "; ".join(f"{k} OI {D.loc[k, 'pct']:+.1f}% price {D.loc[k, 'px']:+.2f}%" for k in ("NIFTY", "BANKNIFTY") if k in D.index)
    print(f"{hm} compared with {prev_t.strftime('%H:%M')} ({mins} min) | stocks {len(stocks)} | median OI change {med_oi:+.2f}% price {med_px:+.2f}% | triggers {len(trig)}")
    if trig.empty:
        if not dry:
            json.dump({**st, "slot": slot}, open(STATE, "w"))
        return
    wl = [x.strip() for x in open(os.path.join(LEARN, "oi_watchlist.txt")).read().split() if x.strip()] if os.path.exists(os.path.join(LEARN, "oi_watchlist.txt")) else []
    fetch_syms = list(dict.fromkeys(list(trig.index[:MAX_CHAINS]) + wl[:MAX_WATCH]))
    chains = {}
    for sym in fetch_syms:
        try:
            exp, spot, C = fetch_chain(sess, sym)
            chains[sym] = (exp, spot, C)
            time.sleep(0.6)
        except Exception as e:
            print("chain failed", sym, str(e)[:50])
    candle = recent_candle_alerts(now)
    lines = [f"OI / OPTIONS 15-minute scan {hm} (OI change vs {prev_t.strftime('%H:%M')}; market median OI {med_oi:+.2f}%, price {med_px:+.2f}%)", idx_txt, ""]
    sig_rows = []
    for sym, r in trig.head(15).iterrows():
        cls = classify(r.adj, r.px)
        meaning, direction = CLASSES[cls]
        icon = "\U0001f534" if direction == "bear" else "\U0001f7e2"
        conf = [c for c in candle.get(sym, []) if c[1] == direction]
        spot_txt = chains[sym][1] if sym in chains else r.underlyingValue
        head = f"{icon} {sym} Rs {spot_txt:,.2f} | OI {r.latestOI:,.0f} {r.pct:+.1f}% in {mins}m ({r.adj:+.1f}% above the market median) | price {r.px:+.2f}% = {cls} ({meaning})"
        lines.append(head)
        copt = ""
        if sym in chains:
            exp, spot, C = chains[sym]
            t, calls, puts = option_lines(now, sym, exp, spot, C)
            lines.append(t)
        if conf:
            lines.append("   CONFLUENCE: candle alert in the same direction: " + ", ".join(f"{c[0]} {c[2]}" for c in conf))
        sig_rows.append([now.strftime("%Y-%m-%d %H:%M"), sym, "OISHIFT", cls, direction, round(r.pct, 2), round(r.adj, 2), round(r.px, 3), round(float(r.underlyingValue), 2), int(bool(conf))])
        done[day + "|" + sym] = time.time()
    lines.append("")
    lines.append("Screen only. Forward test running (oi_scanner.py track).")
    import live_candle_alert as L
    L.send_chunks("", [l for l in lines if l is not None], dry)
    if not dry:
        append(SIGNALS, ["ts", "symbol", "type", "class", "direction", "oi_pct15", "oi_adj", "px15", "price", "confluence"], sig_rows)
        for sym, (exp, spot, C) in chains.items():
            ts = now.strftime("%Y-%m-%d %H:%M")
            append(CHAIN_LOG, ["ts", "symbol", "expiry", "spot", "strike", "ce_oi", "pe_oi", "ce_vol", "pe_vol", "ce_ltp", "pe_ltp", "ce_iv", "pe_iv"],
                   [[ts, sym, exp, spot, r.k, r.ce_oi, r.pe_oi, r.ce_vol, r.pe_vol, r.ce_ltp, r.pe_ltp, r.ce_iv, r.pe_iv] for r in C.itertuples()])
        wlp = os.path.join(LEARN, "oi_watchlist.txt")
        cur_wl = open(wlp).read().split() if os.path.exists(wlp) else []
        add = [s for s in trig.index[:5] if s not in cur_wl]
        if add and len(cur_wl) < 40:
            open(wlp, "w").write("\n".join(cur_wl + add) + "\n")
        json.dump({"slot": slot, "cool": {k: v for k, v in done.items() if k.startswith(day)}}, open(STATE, "w"))


def track(dry=False):
    """Forward test: price move after each OI-shift signal (+15/+30/+60 min, from the 15-minute snapshots) minus the market median."""
    if not os.path.exists(SIGNALS):
        print("no signals logged yet")
        return
    sg = pd.read_csv(SIGNALS)
    o = pd.read_csv(OI_LOG, usecols=["ts", "symbol", "underlying"]).drop_duplicates(["ts", "symbol"])
    o["t"] = pd.to_datetime(o.ts)
    rows = []
    for r in sg.itertuples():
        t0 = pd.Timestamp(r.ts)
        s = o[o.symbol == r.symbol]
        res = {}
        for m in (15, 30, 60):
            tgt = t0 + pd.Timedelta(minutes=m)
            c = s[(s.t >= tgt - pd.Timedelta(minutes=6)) & (s.t <= tgt + pd.Timedelta(minutes=6)) & (s.t.dt.date == t0.date())]
            a = s[(s.t >= t0 - pd.Timedelta(minutes=6)) & (s.t <= t0 + pd.Timedelta(minutes=6))]
            if c.empty or a.empty:
                res[m] = np.nan
                continue
            tc, ta = c.iloc[(c.t - tgt).abs().argsort()[:1]], a.iloc[(a.t - t0).abs().argsort()[:1]]
            mk = o[(o.t == tc.t.iloc[0])].set_index("symbol").underlying / o[(o.t == ta.t.iloc[0])].set_index("symbol").underlying
            mk = (mk.dropna() - 1) * 100
            move = (float(tc.underlying.iloc[0]) / float(ta.underlying.iloc[0]) - 1) * 100
            sgn = -1 if r.direction == "bear" else 1
            res[m] = sgn * (move - float(mk.median()))
        rows.append(dict(cls=r._4, conf=r.confluence, **{f"x{m}": res[m] for m in (15, 30, 60)}))
    R = pd.DataFrame(rows)
    lines = [f"OI-shift forward test: {len(R)} signals logged ({sg.ts.str[:10].nunique()} days). Excess = move in the signal's direction minus the market median move."]
    for name, g in [("ALL", R)] + [(k, v) for k, v in R.groupby("cls")] + [("with candle confluence", R[R.conf == 1])]:
        if len(g) == 0:
            continue
        txt = ", ".join(f"+{m}m {g[f'x{m}'].mean():+.2f}% (n={g[f'x{m}'].notna().sum()}, hit {(g[f'x{m}'] > 0).sum() / max(g[f'x{m}'].notna().sum(), 1) * 100:.0f}%)" for m in (15, 30, 60))
        lines.append(f"  {name}: {txt}")
    out = "\n".join(lines)
    print(out)
    if not dry:
        open(os.path.join(LEARN, "oi_forward.md"), "w").write(out + "\n")


if __name__ == "__main__":
    cmd = next((a for a in sys.argv[1:] if not a.startswith("--")), "scan")
    if cmd == "track":
        track("--dry" in sys.argv)
    else:
        scan("--dry" in sys.argv, "--force" in sys.argv)
