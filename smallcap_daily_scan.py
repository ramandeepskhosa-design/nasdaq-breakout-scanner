#!/usr/bin/env python3
"""Nifty Smallcap 250 daily scan: all six EMAs rising and in bullish order, and the latest
daily candle is a green candle with (almost) no wick on either side.

  EMA 5 > 10 > 20 > 50 > 100 > 200   (stacked)  AND  every EMA higher than the day before (rising)
  latest candle: green, upper wick <= MAX_WICK % and lower wick <= MAX_WICK % of its range
  candle range >= MIN_RANGE % of price (skips tiny candles), 20-day turnover >= MIN_TURNOVER_CR crore rupees

Usage: python3 smallcap_daily_scan.py [--wick=10] [--csv=path]
"""
import io
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

LIST_URL = "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
EMAS = (5, 10, 20, 50, 100, 200)
MAX_WICK = 10.0
MIN_RANGE = 1.5
MIN_TURNOVER_CR = 2.0


def constituents():
    r = requests.get(LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df.rename(columns={"Company Name": "name", "Industry": "industry", "Symbol": "sym"})


def download(tickers):
    out = {}
    for i in range(0, len(tickers), 50):
        part = tickers[i:i + 50]
        d = yf.download(part, period="2y", interval="1d", group_by="ticker",
                        progress=False, auto_adjust=False, threads=False)
        for t in part:
            try:
                s = d[t].dropna(subset=["Open", "High", "Low", "Close"])
                if len(s) > 230:
                    out[t] = s
            except Exception:
                pass
        time.sleep(1)
    return out


def scan(data, max_wick=MAX_WICK):
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    open_now = now.weekday() < 5 and (9, 15) <= (now.hour, now.minute) < (15, 35)
    rows = []
    for t, df in data.items():
        if open_now and df.index[-1].date() == now.date():
            df = df.iloc[:-1]          # today's candle is still forming
        c = df.Close
        e = {p: c.ewm(span=p, adjust=False).mean() for p in EMAS}
        stacked = all(e[a].iloc[-1] > e[b].iloc[-1] for a, b in zip(EMAS, EMAS[1:]))
        rising = all(e[p].iloc[-1] > e[p].iloc[-2] for p in EMAS)
        r = df.iloc[-1]
        rng = r.High - r.Low
        if rng <= 0:
            continue
        green = r.Close > r.Open
        upper = (r.High - r.Close) / rng * 100
        lower = (r.Open - r.Low) / rng * 100
        body = (r.Close - r.Open) / rng * 100
        turnover = (df.Close * df.Volume).tail(20).mean() / 1e7
        rows.append(dict(t=t, date=df.index[-1].date(), close=r.Close, chg=(r.Close / c.iloc[-2] - 1) * 100,
                         body=body, upper=upper, lower=lower, range_pct=rng / r.Close * 100,
                         vol_x=r.Volume / df.Volume.tail(21).iloc[:-1].mean() if df.Volume.tail(21).iloc[:-1].mean() > 0 else np.nan,
                         turnover_cr=turnover, d200=(r.Close / e[200].iloc[-1] - 1) * 100,
                         from_high=(r.Close / df.High.tail(250).max() - 1) * 100,
                         stacked=stacked, rising=rising, green=green))
    R = pd.DataFrame(rows)
    ok = (R.stacked & R.rising & R.green & (R.upper <= max_wick) & (R.lower <= max_wick)
          & (R.range_pct >= MIN_RANGE) & (R.turnover_cr >= MIN_TURNOVER_CR))
    return R, R[ok].sort_values("body", ascending=False)


if __name__ == "__main__":
    wick = next((float(a.split("=")[1]) for a in sys.argv if a.startswith("--wick=")), MAX_WICK)
    cons = constituents()
    cons["t"] = cons.sym + ".NS"
    data = download(list(cons.t))
    R, hits = scan(data, wick)
    hits = hits.merge(cons[["t", "name", "industry"]], on="t", how="left")
    print(f"scanned {len(R)} of {len(cons)} smallcap stocks; latest candle {R.date.max()}")
    print(f"EMAs stacked+rising: {int((R.stacked & R.rising).sum())} | + green candle: {int((R.stacked & R.rising & R.green).sum())} | "
          f"+ wicks <= {wick:.0f}% both sides, range >= {MIN_RANGE}%, turnover >= {MIN_TURNOVER_CR} cr: {len(hits)}")
    for _, r in hits.iterrows():
        print(f"{r.t.replace('.NS', ''):12s} {r['name'][:28]:28s} {r.industry[:18]:18s} close {r.close:9,.1f} {r.chg:+5.1f}% "
              f"body {r.body:3.0f}% wicks {r.upper:2.0f}/{r.lower:2.0f}% vol {r.vol_x:4.1f}x  "
              f"above 200EMA {r.d200:+5.1f}%  from 52w high {r.from_high:+5.1f}%")
    arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--csv=")), None)
    if arg:
        hits.to_csv(arg, index=False)
