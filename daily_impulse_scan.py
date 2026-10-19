#!/usr/bin/env python3
"""Daily 'impulse candle in a bullish EMA stack' scan for NSE (large/mid/small caps) and US stocks.

A stock is flagged after the close when its latest DAILY candle is:
  * green, with BOTH wicks <= WICK % of the candle range (body >= 100 - 2*WICK %)
  * body >= BODY_X times the average body of the previous 20 candles (the Impulse 80/3/20 idea)
  * volume >= VOL_X times its 20-day average
and the daily EMAs are stacked bullish:  5 > 9 > 20 > 50 > 100 > 200.

Backtests (6 years, entry next open): see memory 'live-candle-alert-signal-research'. Edge is small and
carried by a minority of big winners - treat the list as candidates to look at, not as orders.

Runs idempotently (one message per market per session) so launchd can call it several times a day,
including after the Mac wakes from sleep.

Usage: python3 daily_impulse_scan.py [--market=NSE,US] [--dry] [--force]
"""
import io
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import live_candle_alert as L

BASE = os.path.dirname(os.path.abspath(__file__))
SENT = os.path.join(BASE, "learning", "daily_scan_sent.json")
SMALLCAP_URL = "https://niftyindices.com/IndexConstituent/ind_niftysmallcap250list.csv"
EMAS = (5, 9, 20, 50, 100, 200)
WICK, BODY_X, VOL_X = 10.0, 3.0, 1.5
MARKETS = {  # name: (tz, session complete after, currency)
    "NSE": ("Asia/Kolkata", (15, 45), "₹"),
    "US": ("America/New_York", (16, 15), "$"),
}


def universe(name):
    if name == "US":
        return [t for t in L.load_markets()["US"]["tickers"] if not t.startswith("^")], {}
    large = L.load_markets()["NSE"]["tickers"]
    names = {}
    cache = os.path.join(BASE, "learning", "smallcap250.csv")
    text = None
    try:
        r = requests.get(SMALLCAP_URL, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}, timeout=20)
        r.raise_for_status()
        text = r.text
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        open(cache, "w").write(text)             # refresh the local copy whenever the download works
    except Exception as e:
        print("smallcap list download failed, using the saved copy:", e)
        if os.path.exists(cache):
            text = open(cache).read()
    small = []
    if text:
        df = pd.read_csv(io.StringIO(text))
        small = [s + ".NS" for s in df.Symbol]
        names = {s + ".NS": n for s, n in zip(df.Symbol, df["Company Name"])}
    return list(dict.fromkeys(large + small)), names


def download(tickers):
    out = {}
    for i in range(0, len(tickers), 50):
        part = tickers[i:i + 50]
        try:
            d = yf.download(part, period="2y", interval="1d", group_by="ticker",
                            progress=False, auto_adjust=False, threads=False)
        except Exception as e:
            print("download chunk failed:", e)
            continue
        for t in part:
            try:
                s = d[t].dropna(subset=["Open", "High", "Low", "Close"])
                if len(s) > 230:
                    out[t] = s
            except Exception:
                pass
        time.sleep(1)
    return out


def scan(data, tz, complete_after, wick=WICK, body_x=BODY_X, vol_x=VOL_X):
    now = datetime.now(ZoneInfo(tz))
    rows = []
    for t, df in data.items():
        if df.index[-1].date() == now.date() and (now.hour, now.minute) < complete_after:
            df = df.iloc[:-1]                      # today's candle is still forming
        if len(df) < 230 or (now.date() - df.index[-1].date()).days > 4:
            continue
        c = df.Close
        e = {p: c.ewm(span=p, adjust=False).mean() for p in EMAS}
        if not all(e[a].iloc[-1] > e[b].iloc[-1] for a, b in zip(EMAS, EMAS[1:])):
            continue
        r = df.iloc[-1]
        rng = r.High - r.Low
        if rng <= 0 or r.Close <= r.Open:
            continue
        up = (r.High - r.Close) / rng * 100
        dn = (r.Open - r.Low) / rng * 100
        avg_body = (df.Close - df.Open).abs().iloc[-21:-1].mean()
        bx = (r.Close - r.Open) / avg_body if avg_body > 0 else np.nan
        vx = r.Volume / df.Volume.iloc[-21:-1].mean() if df.Volume.iloc[-21:-1].mean() > 0 else np.nan
        if up > wick or dn > wick or not bx >= body_x or not vx >= vol_x:
            continue
        rows.append(dict(t=t, date=str(df.index[-1].date()), close=float(r.Close), chg=(r.Close / c.iloc[-2] - 1) * 100,
                         up=up, dn=dn, bx=bx, vx=vx, d50=(r.Close / e[50].iloc[-1] - 1) * 100,
                         from_high=(r.Close / df.High.tail(250).max() - 1) * 100,
                         turnover_cr=(df.Close * df.Volume).tail(20).mean() / 1e7))
    return pd.DataFrame(rows)


def load_sent():
    try:
        return json.load(open(SENT))
    except Exception:
        return {}


def main():
    dry, force = "--dry" in sys.argv, "--force" in sys.argv
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--market=")), list(MARKETS))
    sent = load_sent()
    for name in only:
        tz, after, cur = MARKETS[name]
        now = datetime.now(ZoneInfo(tz))
        if now.weekday() >= 5 and not force:
            session_done = True          # weekend: the Friday candle is the latest one
        elif (now.hour, now.minute) < after and not force:
            session_done = False
        else:
            session_done = True
        tickers, names = universe(name)
        data = download(tickers)
        if not data:
            print(name, "no data"); continue
        last = max(df.index[-1].date() for df in data.values())
        if not session_done and last == now.date():
            last = max((df.index[-2].date() for df in data.values() if len(df) > 1), default=last)
        key = str(last)
        if sent.get(name) == key and not force:
            print(name, "already sent for", key); continue
        hits = scan(data, tz, after)
        hits = hits[hits.date == key] if len(hits) else hits
        hits = hits.sort_values("vx", ascending=False) if len(hits) else hits
        head = (f"<b>📈 Daily impulse scan - {name} - {key}</b>\n"
                f"Green daily candle, both wicks ≤{WICK:.0f}%, body ≥{BODY_X:.0f}× average, volume ≥{VOL_X}× average, "
                f"EMAs 5>9>20>50>100>200. Scanned {len(data)} stocks: <b>{len(hits)}</b> match.")
        lines = []
        for _, r in hits.iterrows():
            sym = r.t.replace(".NS", "")
            nm = f" {names.get(r.t, '')[:22]}" if names.get(r.t) else ""
            lines.append(f"🟢 <b>{sym}</b>{nm} {cur}{r.close:,.2f} ({r.chg:+.1f}%) wicks {r.up:.0f}/{r.dn:.0f}% "
                         f"body {r.bx:.1f}x vol {r.vx:.1f}x, {r.d50:+.0f}% over 50EMA, {r.from_high:+.0f}% from 52w high")
        if not lines:
            L.send(head + "\nNo stock qualifies today.", dry)
        else:
            for k in range(0, len(lines), 30):
                L.send((head if k == 0 else head.split("\n")[0] + " (cont.)") + "\n" + "\n".join(lines[k:k + 30]), dry)
        if not dry:
            sent[name] = key
            os.makedirs(os.path.dirname(SENT), exist_ok=True)
            json.dump(sent, open(SENT, "w"))
        print(name, key, len(hits), "matches")


if __name__ == "__main__":
    L.raise_fd_limit()
    main()
