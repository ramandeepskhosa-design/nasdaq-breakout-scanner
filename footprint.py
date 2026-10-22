#!/usr/bin/env python3
"""Institutional-footprint check for one NSE stock and time window, from free data.

Usage:  python3 footprint.py SYMBOL START END [--date=YYYY-MM-DD] [--no-options]
  e.g.  python3 footprint.py OBEROIRLTY 10:45 12:45
Prints: large-print minutes (>= 8x the median minute volume) and whether price absorbed or continued after them, signed-volume delta
per window (a rough buy/sell proxy: close vs open of each 1-minute bar), effort-vs-result, and (F&O stocks) the NSE option chain
snapshot: PCR, max pain, biggest OI strikes and today's OI build/unwind, plus the futures OI change from NSE's OI-spurts list.
Free data limits: Yahoo 1-minute volume (no order book), NSE snapshots (no intraday history unless oi_snapshot.py was logging)."""
import os, sys, json
import numpy as np, pandas as pd, yfinance as yf, warnings; warnings.filterwarnings("ignore")
TZ="Asia/Kolkata"
def one(sym,period="5d",interval="1m"):
    x=yf.download(sym,period=period,interval=interval,progress=False,auto_adjust=False)
    if x is None or len(x)==0: return None
    x.columns=x.columns.get_level_values(0); x.index=x.index.tz_convert(TZ); return x
def day(x,d): return x[x.index.strftime("%Y-%m-%d")==d]
def footprint(sym,d,t0,t1,label):
    x=one(sym); o=day(x,d); med=o.Volume[o.Volume>0].median()
    o=o.copy(); sign=np.where(o.Close>o.Open,1,np.where(o.Close<o.Open,-1,np.sign(o.Close.diff().fillna(0)))); o["delta"]=sign*o.Volume; o["hm"]=o.index.strftime("%H:%M")
    print(f"\n#### {label}: institutional-footprint check {t0}-{t1} (median minute volume {med:,.0f})")
    w=o[(o.hm>=t0)&(o.hm<=t1)]
    big=w[w.Volume>=8*med]
    print(f"  large-print minutes (>=8x median): {len(big)}")
    for ts,r in big.iterrows():
        nxt=o[(o.index>ts)&(o.index<=ts+pd.Timedelta(minutes=5))]
        mv=(r.Close/r.Open-1)*100; after=(nxt.Close.iloc[-1]/r.Close-1)*100 if len(nxt) else np.nan
        tag="SELL print, price kept falling" if (r.Close<r.Open and after<-0.03) else "SELL print ABSORBED (price recovered)" if (r.Close<r.Open and after>=0.03) else "BUY print, price kept rising" if (r.Close>r.Open and after>0.03) else "BUY print faded" if r.Close>r.Open else "flat print"
        print(f"    {ts.strftime('%H:%M')} {r.Volume:7,.0f} sh ({r.Volume/med:4.0f}x) minute {mv:+.2f}% -> next 5 min {after:+.2f}%  => {tag}")
    print("  signed-volume delta (close vs open per minute; a rough proxy) by window:")
    for a,b in [(t0,t1)]:
        pass
    edges=sorted(set([t0,t1]+[h for h in ("10:45","11:00","11:15","11:30","12:00","12:15","12:30","12:45","13:00") if t0<=h<=t1]))
    for i in range(len(edges)-1):
        ww=o[(o.hm>=edges[i])&(o.hm<edges[i+1])]
        if len(ww)==0: continue
        print(f"    {edges[i]}-{edges[i+1]}: volume {ww.Volume.sum():7,.0f} ({ww.Volume.sum()/(med*len(ww)):.1f}x normal pace) | delta {ww.delta.sum():+8,.0f} ({ww.delta.sum()/ww.Volume.sum()*100:+.0f}% of volume) | price {(ww.Close.iloc[-1]/ww.Open.iloc[0]-1)*100:+.2f}%")
    return o
def chain(sym):
    import oi_snapshot
    s=oi_snapshot.session()
    exp=s.get(f"https://www.nseindia.com/api/option-chain-contract-info?symbol={sym}",timeout=15).json()["expiryDates"][0]
    v=s.get(f"https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={sym}&expiry={exp}",timeout=20).json(); rec=v["records"]; und=rec["underlyingValue"]; rows=[]
    for r in rec["data"]:
        ce,pe=r.get("CE",{}),r.get("PE",{})
        rows.append(dict(k=r["strikePrice"],ce_oi=ce.get("openInterest",0),ce_chg=ce.get("changeinOpenInterest",0),ce_vol=ce.get("totalTradedVolume",0),ce_iv=ce.get("impliedVolatility",0),ce_ltp=ce.get("lastPrice",0),
                         pe_oi=pe.get("openInterest",0),pe_chg=pe.get("changeinOpenInterest",0),pe_vol=pe.get("totalTradedVolume",0),pe_iv=pe.get("impliedVolatility",0),pe_ltp=pe.get("lastPrice",0)))
    C=pd.DataFrame(rows).sort_values("k"); atm=C.iloc[(C.k-und).abs().argsort()[:1]].iloc[0]
    pcr=C.pe_oi.sum()/max(C.ce_oi.sum(),1); pcr_v=C.pe_vol.sum()/max(C.ce_vol.sum(),1)
    mp=None
    ks=C.k.values; pain=[((np.maximum(ks[i]-ks,0)*C.ce_oi.values).sum()+(np.maximum(ks-ks[i],0)*C.pe_oi.values).sum()) for i in range(len(ks))]; mp=ks[int(np.argmin(pain))]
    print(f"\n#### OPTIONS {sym} (expiry {exp}, snapshot {rec['timestamp']}, spot {und:,.1f})")
    print(f"  PCR (OI) {pcr:.2f} | PCR (traded volume) {pcr_v:.2f} | total call OI {C.ce_oi.sum():,.0f} (chg {C.ce_chg.sum():+,.0f}) | total put OI {C.pe_oi.sum():,.0f} (chg {C.pe_chg.sum():+,.0f}) | max pain {mp}")
    print(f"  ATM strike {atm.k:.0f}: call IV {atm.ce_iv:.1f}%, put IV {atm.pe_iv:.1f}% (put-call IV skew {atm.pe_iv-atm.ce_iv:+.1f})")
    print("  biggest CALL OI (resistance):",", ".join(f"{r.k:.0f} ({r.ce_oi:,.0f}, chg {r.ce_chg:+,.0f})" for _,r in C.nlargest(3,'ce_oi').iterrows()))
    print("  biggest PUT OI (support):   ",", ".join(f"{r.k:.0f} ({r.pe_oi:,.0f}, chg {r.pe_chg:+,.0f})" for _,r in C.nlargest(3,'pe_oi').iterrows()))
    print("  largest OI BUILD today -> calls:",", ".join(f"{r.k:.0f} ({r.ce_chg:+,.0f})" for _,r in C.nlargest(3,'ce_chg').iterrows()),"| puts:",", ".join(f"{r.k:.0f} ({r.pe_chg:+,.0f})" for _,r in C.nlargest(3,'pe_chg').iterrows()))
    print("  largest OI UNWIND -> calls:",", ".join(f"{r.k:.0f} ({r.ce_chg:+,.0f})" for _,r in C.nsmallest(2,'ce_chg').iterrows()),"| puts:",", ".join(f"{r.k:.0f} ({r.pe_chg:+,.0f})" for _,r in C.nsmallest(2,'pe_chg').iterrows()))
    return C


if __name__ == "__main__":
    a=[x for x in sys.argv[1:] if not x.startswith("--")]
    if len(a) < 3:
        sys.exit(__doc__)
    sym,t0,t1=a[0].upper(),a[1],a[2]
    d=next((x.split("=")[1] for x in sys.argv if x.startswith("--date=")), pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d"))
    footprint(sym+".NS",d,t0,t1,sym)
    if "--no-options" not in sys.argv:
        try:
            chain(sym)
            import oi_snapshot
            s=oi_snapshot.session(); j=s.get("https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings",timeout=20).json()
            for r in j.get("data",[]):
                if r["symbol"]==sym: print(f"\n  FUTURES OI: {r['latestOI']:,.0f} vs prev {r['prevOI']:,.0f} ({r['changeInOI']:+,.0f}, {r['avgInOI']:+.1f}%) | spot {r['underlyingValue']}  [price down + OI up = short build-up; price down + OI down = long unwinding; price up + OI up = long build-up; price up + OI down = short covering]")
        except Exception as e:
            print("options not available:", str(e)[:80])
