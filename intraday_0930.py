#!/usr/bin/env python3
"""Intraday 09:30 -> 10:30 trading screen for NSE F&O stocks (a screen + learning loop, never an order).

Morning workflow the user described:
  1. Where is the index heading (Nifty / Bank Nifty at 09:30: gap, first candle, side of VWAP)?
  2. Take the F&O top 20 GAINERS and top 20 LOSERS at 09:30 (move from the previous close, 09:15-09:30 candle included).
  3. Keep the ones with plenty of ROOM, closest to the EMA100, whose previous session looked the part:
        BEAR: yesterday price spent the day below EMA5, EMA9 and EMA20 and all EMAs (5,9,20,50,100,200) were descending
        BULL: the mirror image (above, ascending)
  4. Entry at 09:30, exit (book profit) by 10:30.

Sub-commands:
  python3 intraday_0930.py backtest         replay every day in the 15m history and print which filters mattered
  python3 intraday_0930.py precompute       08:55 job: previous-session state for every stock (light at 09:31)
  python3 intraday_0930.py scan [--dry]     09:32 job: the morning list -> Telegram + learning/morning_log.csv
  python3 intraday_0930.py track [--dry]    10:40 job: measures 09:30 -> 10:30 results of logged picks, updates LEARNINGS.md
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
LEARN = os.path.join(BASE, "learning")
STATE = os.path.join(LEARN, "prev_day_state.json")
MLOG = os.path.join(LEARN, "morning_log.csv")
MRES = os.path.join(LEARN, "morning_results.csv")
SKILL_DIR = os.path.expanduser("~/.claude/skills/intraday-0930-trader")
LEARNMD = os.path.join(SKILL_DIR, "LEARNINGS.md")
TZ = "Asia/Kolkata"
DEAD = {"TATAMOTORS", "LTIM", "ZOMATO", "MCDOWELL-N"}
EMAS = (5, 9, 20, 50, 100, 200)
IDX = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANK NIFTY"}
TOPN = 20
COST = 0.05          # % round trip
SHARE_MIN = 0.90     # share of yesterday's candles that must sit below EMA5/9/20 (bear) or above (bull)
ROOM_MIN = 0.7       # % room to yesterday's low (bear) / high (bull)


def universe():
    return [s + ".NS" for s in L.fo_symbols() if s not in DEAD]


def prep(df):
    b = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(b) < 260:
        return None
    b.index = b.index.tz_convert(TZ)
    b["day"] = b.index.strftime("%Y-%m-%d")
    b["hm"] = b.index.strftime("%H:%M")
    for p in EMAS:
        b[f"e{p}"] = b.Close.ewm(span=p, adjust=False).mean()
    return b


def daily_state(b):
    """Per session: close/high/low, EMAs at the last candle, share of candles below/above EMA5+9+20, EMA order."""
    rows = {}
    for day, g in b.groupby("day"):
        if g.hm.iloc[-1] < "15:15":
            continue
        l = g.iloc[-1]
        e = [l[f"e{p}"] for p in EMAS]
        order = "DOWN" if all(e[i] < e[i + 1] for i in range(5)) else ("UP" if all(e[i] > e[i + 1] for i in range(5)) else "MIX")
        below = ((g.Close < g.e5) & (g.Close < g.e9) & (g.Close < g.e20)).mean()
        above = ((g.Close > g.e5) & (g.Close > g.e9) & (g.Close > g.e20)).mean()
        rows[day] = dict(close=float(l.Close), high=float(g.High.max()), low=float(g.Low.min()), e100=float(l.e100), e50=float(l.e50),
                         order=order, below=float(below), above=float(above))
    return rows


def cand_features(first, st, side_hint=None):
    """Features known at 09:30 for one stock, from its first candle `first` and previous-session state `st`."""
    px = float(first.Close)
    move = (px / st["close"] - 1) * 100
    d100 = (px / st["e100"] - 1) * 100
    room_dn = (px / st["low"] - 1) * 100       # room to fall to yesterday's low (negative = already below it)
    room_up = (st["high"] / px - 1) * 100      # room to rise to yesterday's high
    return dict(px=px, move=move, first_pct=(px / float(first.Open) - 1) * 100, d100=d100, room_dn=room_dn, room_up=room_up,
                prev_below=st["below"], prev_above=st["above"], prev_order=st["order"], f_high=float(first.High), f_low=float(first.Low))


def index_status(b_idx, st_idx, day):
    g = b_idx[b_idx.day == day]
    if g.empty or st_idx is None:
        return None
    f = g.iloc[0]
    move = (float(f.Close) / st_idx["close"] - 1) * 100
    d = "UP" if move >= 0.15 else ("DOWN" if move <= -0.15 else "FLAT")
    return dict(move=move, dir=d, first_pct=(float(f.Close) / float(f.Open) - 1) * 100, gap=(float(f.Open) / st_idx["close"] - 1) * 100)


# ------------------------------------------------------------------------------ backtest
def outcomes(b, i0, side):
    """Entry = open of the 09:30 candle, exit = close of the 10:15 candle (= price at 10:30)."""
    if i0 + 4 >= len(b):
        return None
    seg = b.iloc[i0 + 1:i0 + 5]
    if seg.index[0].strftime("%H:%M") != "09:30" or seg.index[-1].strftime("%H:%M") != "10:15" or seg.day.iloc[0] != seg.day.iloc[-1]:
        return None
    s = -1 if side == "BEAR" else 1
    e, x = float(seg.Open.iloc[0]), float(seg.Close.iloc[-1])
    ret = s * (x / e - 1) * 100 - COST
    first = b.iloc[i0]
    stop = float(first.High) if side == "BEAR" else float(first.Low)      # invalidation = back through the first candle's extreme
    hit = (seg.High.max() >= stop) if side == "BEAR" else (seg.Low.min() <= stop)
    managed = ret
    if hit:
        for _, r in seg.iterrows():
            if (side == "BEAR" and r.High >= stop) or (side == "BULL" and r.Low <= stop):
                managed = s * (stop / e - 1) * 100 - COST
                break
    mfe = (1 - seg.Low.min() / e) * 100 if side == "BEAR" else (seg.High.max() / e - 1) * 100     # best move in the trade's favour
    return dict(ret=ret, managed=managed, stop_hit=int(hit), mfe=max(mfe, 0.0))


def build_history(data, idx_data):
    stocks = {}
    for t, df in data.items():
        b = prep(df)
        if b is not None:
            stocks[t] = (b, daily_state(b))
    idx = {}
    for t, df in idx_data.items():
        b = prep(df)
        if b is not None:
            idx[t] = (b, daily_state(b))
    days = sorted({d for b, _ in stocks.values() for d in b.day.unique()})
    rows = []
    for k, day in enumerate(days):
        per = []
        for t, (b, st) in stocks.items():
            g = b[b.day == day]
            if g.empty or g.hm.iloc[0] != "09:15" or len(g) < 6:
                continue
            prev = [d for d in st if d < day]
            if not prev:
                continue
            i0 = b.index.get_loc(g.index[0])
            per.append((t, b, i0, cand_features(g.iloc[0], st[prev[-1]])))
        if len(per) < 100:
            continue
        nift = None
        if "^NSEI" in idx:
            bi, sti = idx["^NSEI"]
            prev = [d for d in sti if d < day]
            nift = index_status(bi, sti[prev[-1]] if prev else None, day)
        moves = pd.Series({t: f["move"] for t, _, _, f in per})
        losers = set(moves.nsmallest(TOPN).index)
        gainers = set(moves.nlargest(TOPN).index)
        for t, b, i0, f in per:
            side = "BEAR" if t in losers else ("BULL" if t in gainers else None)
            if not side:
                continue
            o = outcomes(b, i0, side)
            if not o:
                continue
            rows.append(dict(day=day, t=t, side=side, idx_dir=nift["dir"] if nift else "NA", idx_move=nift["move"] if nift else np.nan, **f, **o))
    R = pd.DataFrame(rows)
    # derived flags
    R["aligned"] = ((R.side == "BEAR") & (R.idx_dir == "DOWN")) | ((R.side == "BULL") & (R.idx_dir == "UP"))
    R["against"] = ((R.side == "BEAR") & (R.idx_dir == "UP")) | ((R.side == "BULL") & (R.idx_dir == "DOWN"))
    R["prev_ok"] = np.where(R.side == "BEAR", R.prev_below >= SHARE_MIN, R.prev_above >= SHARE_MIN)
    R["order_ok"] = np.where(R.side == "BEAR", R.prev_order == "DOWN", R.prev_order == "UP")
    R["room"] = np.where(R.side == "BEAR", R.room_dn, R.room_up)
    R["room_ok"] = R.room >= ROOM_MIN
    R["ad100"] = R.d100.abs()
    R["near100"] = R.groupby(["day", "side"]).ad100.transform(lambda s: s <= s.median())
    return R


def tstat(x):
    d = x.groupby("day").ret.mean()
    return d.mean() / (d.std() / math.sqrt(len(d))) if len(d) > 5 and d.std() > 0 else float("nan")


def table(R, title):
    print(f"\n=== {title} ===")
    print(f"{'group':58s} {'n':>5s} {'days':>4s} {'win%':>5s} {'avg%':>7s} {'managed%':>8s} {'stop%':>5s} {'t':>6s}  halves")
    days = sorted(R.day.unique())
    h = days[len(days) // 2]

    def row(x, label):
        if len(x) < 25:
            print(f"{label:58s} {len(x):5d}  (too few)")
            return
        print(f"{label:58s} {len(x):5d} {x.day.nunique():4d} {(x.ret > 0).mean() * 100:5.1f} {x.ret.mean():+7.3f} {x.managed.mean():+8.3f} {x.stop_hit.mean() * 100:5.0f} "
              f"{tstat(x):+6.2f}  {x[x.day <= h].ret.mean():+.2f}/{x[x.day > h].ret.mean():+.2f}")
    row(R, "all top-20 movers (no filter)")
    row(R[R.aligned], "index moving the same way (aligned)")
    row(R[R.against], "index moving AGAINST the trade")
    row(R[R.prev_ok], f"yesterday {int(SHARE_MIN * 100)}%+ of candles beyond EMA5/9/20 (right side)")
    row(R[R.order_ok], "yesterday all EMAs in order (desc for bear / asc for bull)")
    row(R[R.prev_ok & R.order_ok], "  + both of the above")
    row(R[R.room_ok], f"room >= {ROOM_MIN}% to yesterday's low/high")
    row(R[R.near100], "closest half to EMA100 (that day, that side)")
    row(R[~R.near100], "farthest half from EMA100")
    row(R[R.prev_ok & R.order_ok & R.room_ok], "  prev-day + EMA order + room")
    row(R[R.prev_ok & R.order_ok & R.room_ok & R.aligned], "  + index aligned   (the full morning checklist)")
    row(R[R.prev_ok & R.order_ok & R.room_ok & R.aligned & R.near100], "  + closest half to EMA100")
    row(R[R.aligned & R.room_ok], "index aligned + room (no prev-day rules)")
    row(R[R.aligned & R.order_ok], "index aligned + EMA order")


def backtest():
    tick = universe()
    data = L.fetch(tick + list(IDX), period="60d")
    idx = {k: data.pop(k) for k in list(IDX) if k in data}
    R = build_history(data, idx)
    os.makedirs(LEARN, exist_ok=True)
    R.to_pickle(os.path.join(LEARN, "morning_backtest.pkl"))
    print(f"history: {R.day.nunique()} days, {len(R)} candidate-trades (top {TOPN} gainers + losers each day)")
    for side in ("BEAR", "BULL"):
        table(R[R.side == side], f"{side}  ({'top 20 LOSERS at 09:30 -> short' if side == 'BEAR' else 'top 20 GAINERS at 09:30 -> long'}), entry 09:30 open, exit 10:30")
    print("\nindex direction mix:", R.drop_duplicates("day").idx_dir.value_counts().to_dict())
    return R


# ------------------------------------------------------------------------------ live: precompute / scan / track
RELAX = 0.5          # live tier threshold: share of yesterday's candles beyond EMA5/9/20


def light(df):
    b = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    b.index = b.index.tz_convert(TZ)
    b["day"] = b.index.strftime("%Y-%m-%d")
    b["hm"] = b.index.strftime("%H:%M")
    return b


def tier_of(side, f, aligned):
    ok = f["prev_order"] == ("DOWN" if side == "BEAR" else "UP") and (f["prev_below"] if side == "BEAR" else f["prev_above"]) >= RELAX
    return "A" if (ok and aligned) else ("B" if ok else "C")


def precompute(before=None):
    data = L.fetch(universe() + list(IDX), period="30d")
    out = {"stocks": {}, "idx": {}}
    asof = ""
    for t, df in data.items():
        b = prep(df)
        if b is None:
            continue
        st = daily_state(b)
        if before:
            st = {k: v for k, v in st.items() if k < before}
        if not st:
            continue
        d = max(st)
        asof = max(asof, d)
        (out["idx"] if t in IDX else out["stocks"])[t] = dict(day=d, **st[d])
    out["asof"] = asof
    json.dump(out, open(STATE, "w"))
    print("precomputed", len(out["stocks"]), "stocks, previous session", asof)


def load_state():
    return json.load(open(STATE))


def fmt_side(rows, side, n=8):
    out = []
    for r in rows[:n]:
        out.append(f"{'SHORT' if side == 'BEAR' else 'LONG '} {r['t'].replace('.NS', '')} {r['px']:,.2f} ({r['move']:+.1f}% vs prev close) | {r['d100']:+.1f}% from EMA100 | "
                   f"yday {'below' if side == 'BEAR' else 'above'} EMA5/9/20 {(r['prev_below'] if side == 'BEAR' else r['prev_above']) * 100:.0f}% of candles, EMAs {r['prev_order']} | "
                   f"room to yday {'low' if side == 'BEAR' else 'high'} {r['room']:+.1f}% | tier {r['tier']}")
    return out


def scan(dry=False, force=False, date=None):
    now = datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(TZ))
    day = date or local.strftime("%Y-%m-%d")
    mark = os.path.join(LEARN, "morning_sent.json")
    try:
        if json.load(open(mark)).get("day") == day and not force:
            print("already sent for", day)
            return
    except Exception:
        pass
    if local.weekday() >= 5 and not force:
        print("weekend")
        return
    stt = load_state() if os.path.exists(STATE) else None
    if not stt or stt.get("asof", "") >= day:
        print("previous-session state missing or stale - recomputing")
        precompute(before=day)
        stt = load_state()
    tick = universe()
    data = L.fetch(tick + list(IDX), period="5d" if date else "1d")
    rows, idxs = [], {}
    for t, df in data.items():
        b = light(L.completed(df, now))
        g = b[b.day == day]
        if g.empty or g.hm.iloc[0] != "09:15":
            continue
        f = g.iloc[0]
        if t in IDX:
            st = stt["idx"].get(t)
            if st:
                move = (float(f.Close) / st["close"] - 1) * 100
                idxs[t] = dict(move=move, first=(float(f.Close) / float(f.Open) - 1) * 100, gap=(float(f.Open) / st["close"] - 1) * 100,
                               dir="UP" if move >= 0.15 else ("DOWN" if move <= -0.15 else "FLAT"))
            continue
        st = stt["stocks"].get(t)
        if st:
            rows.append(dict(t=t, **cand_features(f, st)))
    if not idxs or len(rows) < 100:
        print("not enough data yet (first candle not published?)", len(rows))
        return
    R = pd.DataFrame(rows)
    nd = idxs.get("^NSEI", {}).get("dir", "NA")
    losers = R.nsmallest(TOPN, "move").assign(side="BEAR")
    gainers = R.nlargest(TOPN, "move").assign(side="BULL")
    C = pd.concat([losers, gainers], ignore_index=True)
    C["aligned"] = ((C.side == "BEAR") & (nd == "DOWN")) | ((C.side == "BULL") & (nd == "UP"))
    C["room"] = np.where(C.side == "BEAR", C.room_dn, C.room_up)
    C["tier"] = [tier_of(r.side, r, r.aligned) for _, r in C.iterrows()]
    C["ad100"] = C.d100.abs()
    lines = [f"INTRADAY 09:30 screen - {day}"]
    for k, v in idxs.items():
        lines.append(f"{IDX[k]}: {v['move']:+.2f}% vs prev close (gap {v['gap']:+.2f}%, first candle {v['first']:+.2f}%) -> index {v['dir']}")
    lines.append(f"Index heading: {nd}. Top {TOPN} losers / gainers among {len(R)} F&O stocks. Entry 09:30, book profit by 10:30. Stop reference = first-candle high (short) / low (long).")
    for side, label in (("BEAR", "BEARISH list (top losers)"), ("BULL", "BULLISH list (top gainers)")):
        S = C[C.side == side].sort_values("ad100")
        cnt = S.tier.value_counts().to_dict()
        lines.append(f"\n{label}: tier A {cnt.get('A', 0)} | tier B {cnt.get('B', 0)} | tier C {cnt.get('C', 0)}   (A = yesterday EMAs in order + {int(RELAX * 100)}%+ of candles beyond EMA5/9/20 + index same way; B = same without the index; sorted closest to EMA100 first)")
        top = S[S.tier.isin(["A", "B"])]
        if top.empty:
            lines.append("  none qualify - raw movers closest to EMA100:")
            top = S
        lines += ["  " + x for x in fmt_side(top.to_dict("records"), side)]
    lines.append("\nA screen, not advice. Historical 09:30->10:30 results are in the skill LEARNINGS (no proven edge so far).")
    L.send("\n".join(lines), dry)
    if not dry:
        X = C.copy()
        X["day"] = day
        X["idx_dir"] = nd
        X["idx_move"] = idxs.get("^NSEI", {}).get("move", np.nan)
        X["source"] = "live"
        old = pd.read_csv(MLOG) if os.path.exists(MLOG) else pd.DataFrame()
        if len(old):
            old = old[~((old.day == day) & (old.source == "live"))]
        pd.concat([old, X], ignore_index=True).to_csv(MLOG, index=False)
        json.dump({"day": day}, open(mark, "w"))
    print("scan done:", {s: C[C.side == s].tier.value_counts().to_dict() for s in ("BEAR", "BULL")})


def measure_live():
    if not os.path.exists(MLOG):
        return pd.DataFrame()
    log = pd.read_csv(MLOG)
    done = pd.read_csv(MRES) if os.path.exists(MRES) else pd.DataFrame()
    have = set(zip(done.day, done.t)) if len(done) else set()
    todo = log[[(d, t) not in have for d, t in zip(log.day, log.t)]]
    if todo.empty:
        return done
    now = datetime.now(timezone.utc)
    data = L.fetch(sorted(todo.t.unique()), period="5d")
    rows = []
    for _, r in todo.iterrows():
        if r.t not in data:
            continue
        b = light(L.completed(data[r.t], now))
        g = b[b.day == r.day]
        if g.empty or g.hm.iloc[0] != "09:15":
            continue
        i0 = b.index.get_loc(g.index[0])
        o = outcomes(b, i0, r.side)
        if not o:
            continue
        rows.append(dict(day=r.day, t=r.t, side=r.side, tier=r.tier, aligned=r.aligned, idx_dir=r.idx_dir, move=r.move, d100=r.d100, **o))
    if rows:
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(MRES, index=False)
    return done


def grp(x):
    if len(x) == 0:
        return "no data"
    days = x.day.nunique()
    return f"{len(x)} trades / {days} days | win {(x.ret > 0).mean() * 100:.0f}% | avg {x.ret.mean():+.3f}% | managed {x.managed.mean():+.3f}% | best-move {x.mfe.mean():.2f}%"


def track(dry=False):
    done = measure_live()
    bt = pd.read_pickle(os.path.join(LEARN, "morning_backtest.pkl")) if os.path.exists(os.path.join(LEARN, "morning_backtest.pkl")) else pd.DataFrame()
    lines = []
    if len(bt):
        bt = bt.copy()
        bt["tier"] = [tier_of(r.side, r, r.aligned) for _, r in bt.iterrows()]
        lines += ["### Baseline (replay of the last ~59 sessions, same rules)"]
        for side in ("BEAR", "BULL"):
            for tr in ("ALL", "A", "B"):
                x = bt[(bt.side == side) & ((bt.tier == tr) | (tr == "ALL") | ((tr == "B") & (bt.tier == "A")))] if tr != "ALL" else bt[bt.side == side]
                lines.append(f"- {side} tier {tr}{' (A+B)' if tr == 'B' else ''}: {grp(x)}")
    lines.append("")
    lines.append("### LIVE (out-of-sample, since 22 Oct 2026)")
    if len(done):
        for side in ("BEAR", "BULL"):
            for tr in ("ALL", "A", "B"):
                x = done[(done.side == side) & ((done.tier == tr) | (tr == "ALL") | ((tr == "B") & (done.tier == "A")))] if tr != "ALL" else done[done.side == side]
                lines.append(f"- {side} tier {tr}{' (A+B)' if tr == 'B' else ''}: {grp(x)}")
        last = sorted(done.day.unique())[-1]
        d = done[done.day == last]
        lines += ["", f"Latest day {last}: index {d.idx_dir.iloc[0]}; shorts of top losers {d[d.side == 'BEAR'].ret.mean():+.2f}% avg, longs of top gainers {d[d.side == 'BULL'].ret.mean():+.2f}% avg (09:30->10:30, net of 0.05% cost)"]
    else:
        lines.append("- no live days measured yet")
    text = "\n".join(lines)
    os.makedirs(SKILL_DIR, exist_ok=True)
    A0, A1 = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
    block = f"{A0}\n## Auto-updated facts ({datetime.now(ZoneInfo(TZ)).strftime('%Y-%m-%d %H:%M')} IST, written by intraday_0930.py track - do not edit by hand)\n\n{text}\n{A1}"
    old = open(LEARNMD).read() if os.path.exists(LEARNMD) else ""
    if A0 in old and A1 in old:
        out = old[:old.index(A0)] + block + old[old.index(A1) + len(A1):]
    else:
        out = ("# Intraday 09:30 trader - learnings\n\nThe block between the AUTO markers is rewritten by intraday_0930.py track. Below it: the daily Claude review agent's log and proposals.\n\n" +
               block + "\n\n## Agent review log (newest first)\n\n(none yet)\n\n## Proposals awaiting the user's approval\n\n(none)\n")
    open(LEARNMD, "w").write(out)
    print(text)
    mark = os.path.join(LEARN, "morning_tracked.json")
    try:
        prev = json.load(open(mark)).get("day", "")
    except Exception:
        prev = ""
    if len(done):
        last = sorted(done.day.unique())[-1]
        if last != prev:
            L.send("INTRADAY 09:30->10:30 result\n" + text.split("### LIVE", 1)[-1].strip() if True else text, dry)
            if not dry:
                json.dump({"day": last}, open(mark, "w"))


if __name__ == "__main__":
    L.raise_fd_limit()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        backtest()
    elif cmd == "precompute":
        precompute()
    elif cmd == "scan":
        scan(dry="--dry" in sys.argv, force="--force" in sys.argv, date=next((a.split("=")[1] for a in sys.argv if a.startswith("--date=")), None))
    elif cmd == "track":
        track(dry="--dry" in sys.argv)
