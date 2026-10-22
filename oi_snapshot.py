#!/usr/bin/env python3
"""Every 15 minutes during NSE hours: log futures/options open-interest for ALL F&O stocks (one NSE request) and the full
option chain for a small watchlist, so 'what changed at 11:00' can be answered with OI data later.
  learning/oi_log.csv           ts, symbol, latestOI, prevOI, changeInOI, avgInOI(%), volume, underlyingValue
  learning/chain_log.csv        ts, symbol, expiry, spot, strike, ce_oi, ce_chg, pe_oi, pe_chg, ce_vol, pe_vol, ce_iv, pe_iv
  learning/oi_watchlist.txt     one symbol per line (default: the user's names)
Gated by clock (09:16-15:35 IST weekdays) so launchd can call it every 15 minutes all day."""
import csv, json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
LEARN = os.path.join(BASE, "learning")
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
     "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/option-chain"}


def session():
    s = requests.Session()
    s.headers.update(H)
    s.get("https://www.nseindia.com", timeout=10)
    time.sleep(1)
    s.get("https://www.nseindia.com/option-chain", timeout=10)
    time.sleep(1)
    return s


def watchlist():
    p = os.path.join(LEARN, "oi_watchlist.txt")
    if not os.path.exists(p):
        open(p, "w").write("ANANDRATHI\nOBEROIRLTY\nKEI\nNAUKRI\nHINDCOPPER\nCOCHINSHIP\n")
    return [x.strip() for x in open(p).read().split() if x.strip()]


def append(path, header, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def main():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    hm = now.strftime("%H:%M")
    if "--force" not in sys.argv and (now.weekday() >= 5 or not ("09:16" <= hm <= "15:35")):
        return
    ts = now.strftime("%Y-%m-%d %H:%M")
    s = session()
    j = s.get("https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings", timeout=20).json()
    rows = [[ts, r["symbol"], r["latestOI"], r["prevOI"], r["changeInOI"], r["avgInOI"], r["volume"], r["underlyingValue"]] for r in j.get("data", [])]
    os.makedirs(LEARN, exist_ok=True)
    append(os.path.join(LEARN, "oi_log.csv"), ["ts", "symbol", "latestOI", "prevOI", "changeInOI", "avgInOIpct", "volume", "underlying"], rows)
    crow = []
    for sym in watchlist():
        try:
            info = s.get(f"https://www.nseindia.com/api/option-chain-contract-info?symbol={sym}", timeout=15).json()
            exp = info["expiryDates"][0]
            v = s.get(f"https://www.nseindia.com/api/option-chain-v3?type=Equity&symbol={sym}&expiry={exp}", timeout=20).json()["records"]
            for r in v["data"]:
                ce, pe = r.get("CE", {}), r.get("PE", {})
                crow.append([ts, sym, exp, v["underlyingValue"], r["strikePrice"], ce.get("openInterest", 0), ce.get("changeinOpenInterest", 0), pe.get("openInterest", 0),
                             pe.get("changeinOpenInterest", 0), ce.get("totalTradedVolume", 0), pe.get("totalTradedVolume", 0), ce.get("impliedVolatility", 0), pe.get("impliedVolatility", 0)])
        except Exception as e:
            print("chain failed", sym, str(e)[:60])
    if crow:
        append(os.path.join(LEARN, "chain_log.csv"), ["ts", "symbol", "expiry", "spot", "strike", "ce_oi", "ce_chg", "pe_oi", "pe_chg", "ce_vol", "pe_vol", "ce_iv", "pe_iv"], crow)
    print(ts, "logged", len(rows), "OI rows,", len(crow), "chain rows")


if __name__ == "__main__":
    main()
