#!/usr/bin/env python3
"""What changed in an NSE stock at HH:MM? 15m indicator panel + peer/index relative strength + 1-minute volume footprint + OI/options.

Usage: python3 whatchanged.py SYMBOL HH:MM [--date=YYYY-MM-DD] [--peers=TCS,INFY] [--index=^CNXIT]
Shows the 15-minute candles from 45 min before to 60 min after HH:MM with VWAP side, EMA5/9/20 side, RSI14, MACD histogram,
Bollinger %B, volume vs the same slot, Nifty/sector move, then the 1-minute large prints (vs the same clock minute on prior
sessions) and signed-volume delta, relative strength vs index and peers, daily context, and (F&O) NSE OI and option snapshot."""
import os, sys, json
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
import live_candle_alert as L

TZ = "Asia/Kolkata"


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def one(sym, interval="1m", period="7d"):
    x = yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=False)
    if x is None or len(x) == 0:
        return None
    x.columns = x.columns.get_level_values(0)
    x.index = x.index.tz_convert(TZ)
    x["day"] = x.index.strftime("%Y-%m-%d")
    x["hm"] = x.index.strftime("%H:%M")
    return x


def main():
    sym = sys.argv[1].upper()
    hm = sys.argv[2]
    stock = sym + ".NS"
    idx = arg("index", "^NSEI")
    peers = [p + ".NS" for p in arg("peers", "").split(",") if p]
    data = L.fetch([stock, "^NSEI", idx] + peers, period="30d")
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    def prep(df):
        b = L.completed(df.dropna(subset=["Open", "High", "Low", "Close"]), now).copy()
        b.index = b.index.tz_convert(TZ)
        b["day"] = b.index.strftime("%Y-%m-%d")
        b["hm"] = b.index.strftime("%H:%M")
        return b
    b = prep(data[stock])
    D = arg("date") or max(d for d in b.day.unique() if (b[b.day == d].hm == hm).any())
    for p in (5, 9, 20, 50, 100, 200):
        b[f"e{p}"] = b.Close.ewm(span=p, adjust=False).mean()
    tp = (b.High + b.Low + b.Close) / 3
    b["vwap"] = (tp * b.Volume).groupby(b.day).cumsum() / b.Volume.groupby(b.day).cumsum()
    d = b.Close.diff(); up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    b["rsi"] = 100 - 100 / (1 + up / dn)
    m = b.Close.ewm(span=12, adjust=False).mean() - b.Close.ewm(span=26, adjust=False).mean(); b["mh"] = m - m.ewm(span=9, adjust=False).mean()
    sma = b.Close.rolling(20).mean(); sd = b.Close.rolling(20).std(); b["bb"] = (b.Close - (sma - 2 * sd)) / (4 * sd) * 100
    b["slot"] = b.groupby("hm").Volume.transform(lambda x: x.shift().rolling(10, min_periods=5).mean()); b["vr"] = b.Volume / b.slot
    t = b[b.day == D]; prev = b[b.day < D]; pc = float(prev.Close.iloc[-1]); phi = float(prev[prev.day == prev.day.iloc[-1]].High.max()); plo = float(prev[prev.day == prev.day.iloc[-1]].Low.min())
    print(f"{sym} on {D} | prev close {pc:,.2f} (prev-day high {phi:,.2f}, low {plo:,.2f}) | open {t.Open.iloc[0]:,.2f} ({(t.Open.iloc[0] / pc - 1) * 100:+.2f}% gap) | day {(t.Close.iloc[-1] / pc - 1) * 100:+.2f}% | high {t.High.max():,.2f} ({t.High.idxmax().strftime('%H:%M')}) low {t.Low.min():,.2f} ({t.Low.idxmin().strftime('%H:%M')})")
    ni, ix = prep(data["^NSEI"]), prep(data[idx])
    lo = (pd.Timestamp(f"{D} {hm}", tz=TZ) - pd.Timedelta(minutes=60)).strftime("%H:%M"); hi = (pd.Timestamp(f"{D} {hm}", tz=TZ) + pd.Timedelta(minutes=75)).strftime("%H:%M")
    print(f"\n15m candles {lo}-{hi}:  time    O      H      L      C    chg  body upW loW  vol-x  VWAP-side | RSI MACDh BB% | EMA5/9/20 | Nifty  {idx}")
    for ts, r in t[(t.hm >= lo) & (t.hm <= hi)].iterrows():
        rg = r.High - r.Low; u = (r.High - max(r.Open, r.Close)) / rg * 100 if rg else 0; l_ = (min(r.Open, r.Close) - r.Low) / rg * 100 if rg else 0
        n = ni[ni.index == ts]; q = ix[ix.index == ts]
        em = "above all" if r.Close > max(r.e5, r.e9, r.e20) else "below all" if r.Close < min(r.e5, r.e9, r.e20) else "mixed"
        vr = 0 if r.vr != r.vr else r.vr
        mark = "  <==" if r.hm == hm else ""
        print(f"   {r.hm} {r.Open:8.1f}{r.High:7.1f}{r.Low:7.1f}{r.Close:7.1f} {(r.Close / r.Open - 1) * 100:+5.2f}% {abs(r.Close - r.Open) / rg * 100 if rg else 0:3.0f} {u:3.0f} {l_:3.0f} {vr:5.1f}x {'above' if r.Close > r.vwap else 'below'}  | {r.rsi:3.0f} {r.mh:+6.2f} {r.bb:4.0f} | {em:9s} | {(n.Close.iloc[0] / n.Open.iloc[0] - 1) * 100 if len(n) else float('nan'):+.2f} {(q.Close.iloc[0] / q.Open.iloc[0] - 1) * 100 if len(q) else float('nan'):+.2f}{mark}")
    r = t[t.hm == hm].iloc[0]
    print(f"\nat the {hm} close: price {r.Close:,.2f} | VWAP {r.vwap:,.2f} | EMA5/9/20/50/100/200 " + "/".join(f"{r[f'e{p}']:,.0f}" for p in (5, 9, 20, 50, 100, 200)) + f" | vs EMA50 {(r.Close / r.e50 - 1) * 100:+.2f}% EMA100 {(r.Close / r.e100 - 1) * 100:+.2f}% EMA200 {(r.Close / r.e200 - 1) * 100:+.2f}%")
    orr = t[t.hm < "09:45"]; print(f"opening range {orr.Low.min():,.2f}-{orr.High.max():,.2f}: price {'above' if r.Close > orr.High.max() else 'below' if r.Close < orr.Low.min() else 'inside'}; prev-day high/low: {'above high' if r.Close > phi else 'below low' if r.Close < plo else 'inside'}")
    # 1-minute footprint
    x = one(stock)
    if x is not None:
        o = x[x.day == D].copy(); prevs = [dd for dd in x.day.unique() if dd < D]
        base = x[x.day.isin(prevs)].groupby("hm").Volume.median(); o["x"] = o.Volume / o.hm.map(base).replace(0, np.nan)
        med = o.Volume[o.Volume > 0].median(); o["delta"] = np.where(o.Close > o.Open, 1, np.where(o.Close < o.Open, -1, np.sign(o.Close.diff().fillna(0)))) * o.Volume
        t0 = (pd.Timestamp(f"{D} {hm}", tz=TZ) - pd.Timedelta(minutes=15)).strftime("%H:%M"); t1 = (pd.Timestamp(f"{D} {hm}", tz=TZ) + pd.Timedelta(minutes=30)).strftime("%H:%M")
        w = o[(o.hm >= t0) & (o.hm <= t1)]
        print(f"\n1-minute volume {t0}-{t1}: {w.Volume.sum():,.0f} shares = {w.Volume.sum() / o.hm.map(base)[w.index].sum():.1f}x the usual for these minutes | delta {w.delta.sum():+,.0f} ({w.delta.sum() / max(w.Volume.sum(), 1) * 100:+.0f}%) | median minute today {med:,.0f}")
        for ts, rr in w[(w.Volume >= 4 * med)].iterrows():
            nxt = o[(o.index > ts) & (o.index <= ts + pd.Timedelta(minutes=5))]; aft = (nxt.Close.iloc[-1] / rr.Close - 1) * 100 if len(nxt) else float("nan")
            mv = (rr.Close / rr.Open - 1) * 100
            tag = "sell pressure, kept falling" if (mv < 0 and aft < -0.03) else "sell ABSORBED" if mv < 0 and aft >= 0.03 else "buy, kept rising" if (mv > 0 and aft > 0.03) else "buy faded" if mv > 0 else "flat (heavy volume, no move)"
            print(f"   {ts.strftime('%H:%M')} {rr.Volume:8,.0f} sh ({0 if rr.x != rr.x else rr.x:5.1f}x same minute, value Rs {rr.Volume * rr.Close / 1e7:5.2f} cr) minute {mv:+.2f}% next5 {aft:+.2f}% -> {tag}")
    # relative strength
    print("\nrelative strength (cumulative % since open): stock | Nifty | " + idx + " | peers")
    def cum(df, hh):
        y = df[df.day == D]; z = y[y.hm == hh]
        return (float(z.Close.iloc[0]) / float(y.Open.iloc[0]) - 1) * 100 if len(z) else float("nan")
    pb = {p: prep(data[p]) for p in peers if p in data}
    t_hm = [(pd.Timestamp(f"{D} {hm}", tz=TZ) + pd.Timedelta(minutes=k)).strftime("%H:%M") for k in (-60, -30, -15, 0, 15, 30, 60)]
    for hh in t_hm:
        print(f"  {hh}: {cum(b, hh):+.2f}% | {cum(ni, hh):+.2f}% | {cum(ix, hh):+.2f}%" + "".join(f" | {p.replace('.NS', '')} {cum(pb[p], hh):+.2f}%" for p in pb))
    # daily
    dd = yf.download(stock, period="1y", interval="1d", progress=False, auto_adjust=False); dd.columns = dd.columns.get_level_values(0)
    for p in (20, 50, 100, 200):
        dd[f"e{p}"] = dd.Close.ewm(span=p, adjust=False).mean()
    c = dd.Close; dl = c.diff(); u2 = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); d2 = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean(); rs_ = 100 - 100 / (1 + u2 / d2); lz = dd.iloc[-1]
    print(f"\nDAILY: close {lz.Close:,.1f} | EMA20/50/100/200 {lz.e20:,.0f}/{lz.e50:,.0f}/{lz.e100:,.0f}/{lz.e200:,.0f} | RSI14 {rs_.iloc[-1]:.0f} | 52w high {dd.High.tail(250).max():,.0f} ({(lz.Close / dd.High.tail(250).max() - 1) * 100:+.1f}%) | last closes {[round(float(v), 1) for v in c.tail(5)]}")
    if "--no-options" not in sys.argv:
        try:
            import oi_snapshot
            s = oi_snapshot.session()
            for r_ in s.get("https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings", timeout=15).json().get("data", []):
                if r_["symbol"] == sym:
                    print(f"\nFUTURES OI now: {r_['latestOI']:,.0f} vs prev close {r_['prevOI']:,.0f} ({r_['changeInOI']:+,.0f}, {r_['avgInOI']:+.1f}%)")
            if os.path.exists(os.path.join("learning", "oi_log.csv")):
                lg = pd.read_csv(os.path.join("learning", "oi_log.csv")); lg = lg[(lg.symbol == sym) & (lg.ts.str[:10] == D)]
                if len(lg) > 1:
                    print("logged OI snapshots today:", ", ".join(f"{a[11:]} {int(v):,}" for a, v in zip(lg.ts, lg.latestOI)))
                else:
                    print("(no intraday OI history for that day - the logger started 22 Oct 15:24)")
        except Exception as e:
            print("OI lookup failed:", str(e)[:80])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main()
