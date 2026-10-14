#!/usr/bin/env python3
"""Nightly learning loop for the NSE live candle alerts.

Every run it:
  1. Downloads the last 60 days of 15m candles (Yahoo) for the NSE universe + Nifty.
  2. Replays the exact live alert rules bar by bar (so Mac sleep never leaves gaps) and
     appends every NEW day's signals, with outcomes, to learning/signals.csv.gz.
  3. Scores a grid of candidate "setups" (alert type x direction x time slot x Nifty
     confirmation x EMA alignment) with day-clustered statistics, split-half and recent
     checks, and a strict multiple-testing threshold.
  4. Writes learning/scorecard.json (live alerts read it to tag PROVEN setups) and
     sends a Telegram summary: today's winners/losers + which setups are proven.

Outcome of a signal: enter at the NEXT candle open, exit at the day close, minus 0.05% cost.
(R-multiple: +1.5 ATR target vs -1.5 ATR stop within the day.)

Usage:  python3 signal_learning.py [--dry] [--no-send] [--rebuild]
"""
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import live_candle_alert as L

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "learning")
STORE = os.path.join(OUT_DIR, "signals.csv.gz")
SCORECARD = os.path.join(OUT_DIR, "scorecard.json")
TZ = "Asia/Kolkata"
COST = 0.05          # % round trip
MIN_N, MIN_DAYS = 150, 20
T_PROVEN, T_PROMISING = 3.5, 2.0   # strict because ~290 setups are tested at once


# ----------------------------------------------------------------------------- data
def download(tickers):
    data = {}
    for i in range(0, len(tickers), 25):
        part = tickers[i:i + 25]
        d = yf.download(part, period="60d", interval="15m", group_by="ticker",
                        progress=False, auto_adjust=False, threads=False)
        for t in part:
            try:
                s = L.flatten(d[t]).dropna(subset=["Open", "High", "Low", "Close"])
                if len(s) > 500:
                    s = s.copy()
                    s.index = s.index.tz_convert(TZ)
                    data[t] = s
            except Exception:
                pass
        time.sleep(1)
    return data


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_rows(data, nifty, only_dates):
    nifty = nifty.copy()
    nifty["date"] = nifty.index.strftime("%Y-%m-%d")
    nifty["n_chg"] = (nifty.Close / nifty.groupby("date").Open.transform("first") - 1) * 100
    rows = []
    for t, df in data.items():
        df = df.copy()
        df["date"] = df.index.strftime("%Y-%m-%d")
        df["mday"] = df.index.hour * 60 + df.index.minute
        g = df.groupby("date")
        c = df.Close
        pc = c.shift()
        tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
        df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
        df["rsi"] = rsi(c)
        for p in L.EMA_PERIODS:
            df[f"e{p}"] = c.ewm(span=p, adjust=False).mean()
        df["bull_al"] = ((df.e5 > df.e10) & (df.e10 > df.e20) & (df.e20 > df.e50) & (df.e50 > df.e100) & (df.e100 > df.e200)).astype(int)
        df["bear_al"] = ((df.e5 < df.e10) & (df.e10 < df.e20) & (df.e20 < df.e50) & (df.e50 < df.e100) & (df.e100 < df.e200)).astype(int)
        tp = (df.High + df.Low + df.Close) / 3
        vol = df.Volume.replace(0, np.nan)
        df["vwap"] = (tp * vol).groupby(df.date).cumsum() / vol.groupby(df.date).cumsum()
        df["vol_ratio"] = df.Volume / df.Volume.rolling(20).mean()
        df["chg_open"] = (c / g.Open.transform("first") - 1) * 100
        df["cumhi"], df["cumlo"] = g.High.cummax(), g.Low.cummin()
        df["nchg"] = nifty.n_chg.reindex(df.index)
        n = len(df)
        daypos = g.cumcount().values
        daylen = g.Close.transform("count").values
        O, H, Lw, C = df.Open.values, df.High.values, df.Low.values, df.Close.values
        dates = df.date.values
        fired = {}
        for pos in range(60, n - 1):
            day = dates[pos]
            ts = df.index[pos]
            if (ts.hour, ts.minute) == (15, 15) or daypos[pos] + 1 >= daylen[pos]:
                continue
            if day not in only_dates:
                continue
            bars = df.iloc[max(0, pos - 80):pos + 1][["Open", "High", "Low", "Close", "Volume"]]
            for kind, d, _ in L.evaluate(bars) + L.evaluate_extreme(bars, TZ):
                if kind == "EXTREME":          # same cool-down as the live service (60 min = 4 candles)
                    if pos - fired.get((day, d), -99) < 4:
                        continue
                    fired[(day, d)] = pos
                s = 1 if d == "bull" else -1
                r = df.iloc[pos]
                rng = r.High - r.Low
                red = r.Close < r.Open
                wick = ((r.Close - r.Low) if red else (r.High - r.Close)) / rng * 100 if rng > 0 else 0
                body = abs(r.Close - r.Open) / rng * 100 if rng > 0 else 0
                ext = (r.Close - r.cumlo) / r.Close * 100 if s == -1 else (r.cumhi - r.Close) / r.Close * 100
                e = O[pos + 1]
                R = 1.5 * r.atr / e
                end = pos + 1
                while end + 1 < n and daypos[end + 1] > daypos[end]:
                    end += 1
                tgt, stp, res = e * (1 + s * R), e * (1 - s * R), None
                for j in range(pos + 1, end + 1):
                    t_hit = (Lw[j] <= tgt) if s == -1 else (H[j] >= tgt)
                    s_hit = (H[j] >= stp) if s == -1 else (Lw[j] <= stp)
                    if s_hit:
                        res = -1.0
                        break
                    if t_hit:
                        res = 1.0
                        break
                eod = (C[end] - e) / e * s * 100
                if res is None:
                    res = float(np.clip(eod / 100 / R, -1, 1))
                rows.append(dict(
                    t=t, sym=t.replace(".NS", ""), date=day, ts=ts.isoformat(), mday=int(r.mday),
                    kind=kind, dir=d, wick=round(wick, 1), body=round(body, 1), ext_dist=round(ext, 3),
                    n_chg_dir=round(float(r.nchg) * s, 3) if pd.notna(r.nchg) else np.nan,
                    chg_open_dir=round(r.chg_open * s, 3), rsi_dir=round(r.rsi if s == 1 else 100 - r.rsi, 1),
                    vol_ratio=round(float(r.vol_ratio), 2) if pd.notna(r.vol_ratio) else np.nan,
                    aligned=int(r.bull_al if s == 1 else r.bear_al),
                    entry=round(float(e), 2), ret_eod=round(eod - COST, 4), r_mult=round(res, 3)))
    out = pd.DataFrame(rows)
    if len(out):
        out["breadth"] = out.groupby(["ts", "dir"]).t.transform("nunique")
    return out


def update_store(rebuild=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    old = pd.read_csv(STORE) if (os.path.exists(STORE) and not rebuild) else pd.DataFrame()
    data = download(L.load_markets()["NSE"]["tickers"] + ["^NSEI"])
    nifty = data.pop("^NSEI")
    now = pd.Timestamp.now(tz=TZ)
    all_dates = sorted({d for df in data.values() for d in df.index.strftime("%Y-%m-%d").unique()})
    # a day is complete once the 15:15 candle has closed (15:30) - never learn from a partial day
    complete = [d for d in all_dates if d < now.strftime("%Y-%m-%d") or (now.hour, now.minute) >= (15, 35)]
    if len(old):
        last = old.date.max()
        todo = {d for d in complete if d >= last}      # refresh the newest stored day + anything newer
        old = old[~old.date.isin(todo)]
    else:
        todo = set(complete)
    new = build_rows(data, nifty, todo)
    store = pd.concat([old, new], ignore_index=True).sort_values(["date", "ts", "t"]).reset_index(drop=True)
    store.to_csv(STORE, index=False, compression="gzip")
    return store, len(new)


# ----------------------------------------------------------------------- scorecard
SLOTS = {"all": (0, 2000), "09:15-10:00": (555, 600), "10:00-11:30": (600, 690),
         "11:30-13:00": (690, 780), "13:00-15:00": (780, 900)}
NIFTY = {"any": None, "nifty with >0.2%": 0.2, "nifty with >0.4%": 0.4}


def setup_mask(d, kind, direction, slot, nth, aligned):
    m = pd.Series(True, index=d.index)
    if kind != "ALL":
        m &= d.kind == kind
    if direction != "both":
        m &= d.dir == direction
    lo, hi = SLOTS[slot]
    m &= (d.mday >= lo) & (d.mday < hi)
    if NIFTY[nth] is not None:
        m &= d.n_chg_dir > NIFTY[nth]
    if aligned:
        m &= d.aligned == 1
    return m


def day_t(x, key):
    dd = x.groupby(key).ret_eod.mean()
    if len(dd) < 3 or dd.std() == 0:
        return 0.0, len(dd)
    return float(dd.mean() / (dd.std() / np.sqrt(len(dd)))), len(dd)


def score_setups(d):
    dates = sorted(d.date.unique())
    half = dates[len(dates) // 2]
    recent = set(dates[-20:])
    res = []
    for kind in ["ALL", "EXTREME", "IMPULSE", "NOWICK"]:
        for direction in ["both", "bear", "bull"]:
            for slot in SLOTS:
                for nth in NIFTY:
                    for aligned in (False, True):
                        x = d[setup_mask(d, kind, direction, slot, nth, aligned)]
                        if len(x) < 30:
                            continue
                        t, days = day_t(x, "date")
                        h1, h2 = x[x.date <= half].ret_eod.mean(), x[x.date > half].ret_eod.mean()
                        rec = x[x.date.isin(recent)]
                        res.append(dict(kind=kind, dir=direction, slot=slot, nifty=nth, aligned=aligned,
                                        n=len(x), days=days, win=round((x.ret_eod > 0).mean() * 100, 1),
                                        ret=round(x.ret_eod.mean(), 3), expR=round(x.r_mult.mean(), 3),
                                        t=round(t, 2), half1=round(h1, 3), half2=round(h2, 3),
                                        recent_n=len(rec), recent=round(rec.ret_eod.mean(), 3) if len(rec) else None))
    out = pd.DataFrame(res)
    ok = (out.n >= MIN_N) & (out.days >= MIN_DAYS)
    out["status"] = "no edge"
    out.loc[(out.t >= T_PROMISING) & (out.ret > 0) & ~ok, "status"] = "needs more data"
    both_pos = (out.half1 > 0) & (out.half2 > 0) & (out.recent.fillna(-1) > 0)
    out.loc[ok & (out.t >= T_PROMISING) & (out.ret > 0), "status"] = "promising"
    out.loc[ok & (out.t >= T_PROVEN) & (out.ret > 0) & both_pos, "status"] = "PROVEN"
    out.loc[ok & (out.t <= -T_PROMISING) & (out.ret < 0), "status"] = "losing"
    return out.sort_values("t", ascending=False).reset_index(drop=True)


def write_scorecard(d, sc):
    proven = sc[sc.status == "PROVEN"]
    card = {"updated": datetime.now().isoformat(timespec="seconds"), "signals": int(len(d)),
            "days": int(d.date.nunique()), "through": d.date.max(),
            "thresholds": {"proven_t": T_PROVEN, "min_n": MIN_N, "min_days": MIN_DAYS},
            "proven": [{**r, "slot_range": list(SLOTS[r["slot"]]), "nifty_min": NIFTY[r["nifty"]]}
                       for r in proven.to_dict("records")],
            "top": sc.head(10).to_dict("records")}
    json.dump(card, open(SCORECARD, "w"), indent=1, default=str)
    return card


# --------------------------------------------------------------------------- report
def fmt_setup(r):
    bits = [r["kind"], r["dir"]]
    if r["slot"] != "all":
        bits.append(r["slot"])
    if r["nifty"] != "any":
        bits.append(r["nifty"])
    if r["aligned"]:
        bits.append("EMA-aligned")
    return " / ".join(bits)


def build_report(store, sc, card, new_rows):
    last = store.date.max()
    day_all = store[store.date == last]
    td = day_all[day_all.aligned == 1]          # what the live service actually alerts (EMA-aligned only)
    lines = [f"🧠 <b>Daily learning - NSE - {last}</b>",
             f"Memory: {len(store):,} signals over {store.date.nunique()} days "
             f"(+{new_rows} newly learned)", ""]
    if len(td):
        lines.append(f"<b>{last} alerts you received:</b> {len(td)}, {(td.ret_eod > 0).mean() * 100:.0f}% profitable by the close "
                     f"(entry next candle, after 0.05% cost), average {td.ret_eod.mean():+.2f}%")
        lines.append(f"(all {len(day_all)} raw signals incl. non-EMA-aligned: {(day_all.ret_eod > 0).mean() * 100:.0f}% won, {day_all.ret_eod.mean():+.2f}%)")
        for k, g in td.groupby(["kind", "dir"]):
            lines.append(f"  • {L.kind_label(k[0], k[1])} {k[1]}: {len(g)} → {(g.ret_eod > 0).mean() * 100:.0f}% won, {g.ret_eod.mean():+.2f}%")
        lines.append("")
        w = td.sort_values("ret_eod", ascending=False)
        lines.append("<b>Best trades</b>")
        for _, r in w.head(5).iterrows():
            lines.append(f"✅ {r.sym} {r.dir} {r.kind} {pd.Timestamp(r.ts).strftime('%H:%M')} → {r.ret_eod:+.2f}% "
                         f"(Nifty {r.n_chg_dir:+.2f}% {'with' if r.n_chg_dir > 0 else 'against'}, body {r.body:.0f}%, wick {r.wick:.0f}%)")
        lines.append("<b>Worst trades</b>")
        for _, r in w.tail(5).iloc[::-1].iterrows():
            lines.append(f"❌ {r.sym} {r.dir} {r.kind} {pd.Timestamp(r.ts).strftime('%H:%M')} → {r.ret_eod:+.2f}% "
                         f"(Nifty {r.n_chg_dir:+.2f}% {'with' if r.n_chg_dir > 0 else 'against'}, body {r.body:.0f}%, wick {r.wick:.0f}%)")
        wn = td[td.ret_eod > 0]
        ls = td[td.ret_eod <= 0]
        if len(wn) and len(ls):
            lines.append(f"Winners vs losers today: Nifty-with-trade {wn.n_chg_dir.mean():+.2f}% vs {ls.n_chg_dir.mean():+.2f}%, "
                         f"signal time {int(wn.mday.mean()) // 60:02d}:{int(wn.mday.mean()) % 60:02d} vs "
                         f"{int(ls.mday.mean()) // 60:02d}:{int(ls.mday.mean()) % 60:02d}")
        lines.append("")
    n_proven = int((sc.status == "PROVEN").sum())
    n_prom = int((sc.status == "promising").sum())
    lines.append(f"<b>Setups tested: {len(sc)}</b> → ⭐ proven {n_proven}, promising {n_prom}, need more data {(sc.status == 'needs more data').sum()}, losing {(sc.status == 'losing').sum()}")
    lines.append("(proven = t≥3.5, ≥150 trades, ≥20 days, profitable in both halves and the last 20 days)")
    show = sc[sc.status.isin(["PROVEN", "promising", "needs more data"])].head(5)
    if show.empty:
        lines.append("No setup has proven a real edge yet. Best so far:")
        show = sc.head(3)
    for _, r in show.iterrows():
        lines.append(f"{'⭐' if r.status == 'PROVEN' else '•'} {fmt_setup(r)}: {r.n} trades, {r.win}% won, "
                     f"{r.ret:+.3f}%/trade, t={r.t} [{r.status}]")
    bad = sc[sc.status == "losing"].sort_values("t").head(3)
    for _, r in bad.iterrows():
        lines.append(f"🚫 avoid {fmt_setup(r)}: {r.n} trades, {r.ret:+.3f}%/trade, t={r.t}")
    return lines


def main():
    dry = "--dry" in sys.argv
    store, new_rows = update_store(rebuild="--rebuild" in sys.argv)
    sc = score_setups(store)
    card = write_scorecard(store, sc)
    lines = build_report(store, sc, card, new_rows)
    text = "\n".join(lines)
    print(text)
    marker = os.path.join(OUT_DIR, "last_sent.txt")
    last_sent = open(marker).read().strip() if os.path.exists(marker) else ""
    day = store.date.max()
    if "--no-send" in sys.argv or dry:
        return
    if last_sent == day and "--force" not in sys.argv:
        print(f"report for {day} already sent - skipping")      # the 15:45 run and the later backups share this marker
        return
    for k in range(0, len(lines), 40):
        L.send("\n".join(lines[k:k + 40]), False)
    open(marker, "w").write(day)      # only after every part was sent


if __name__ == "__main__":
    L.raise_fd_limit()
    main()
