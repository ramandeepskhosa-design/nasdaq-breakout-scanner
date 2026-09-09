"""
Options ATM Crossover Scanner (15-min chart, EMA100 / EMA200) → Telegram
==========================================================================
Watches NIFTY, BANKNIFTY, and F&O-eligible NSE stocks (approximated here
by Nifty 50 + Nifty Next 50 + Midcap 50) for the underlying's 15-min
price crossing its EMA100 or EMA200.

On a cross, reports the ATM (at-the-money) strike and direction (CE for
a bullish cross above, PE for a bearish cross below) so you know which
option to look at. Free Yahoo Finance data has no live NSE options chain,
so this gives strike + direction only — check the actual premium on
Kite/your broker app.
"""
import json
import os
import time
import urllib.request
import urllib.parse

import yfinance as yf
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "nse_index_stocks.json")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

# yfinance ticker -> (display name, strike interval, is index)
INDEX_INSTRUMENTS = {
    "^NSEI":    ("NIFTY", 50, True),
    "^NSEBANK": ("BANKNIFTY", 100, True),
}


def load_indices():
    with open(INDEX_FILE) as f:
        return json.load(f)


def strike_interval_for_price(price):
    """Rough ATM strike-rounding interval based on price magnitude.
    NSE's real per-stock strike gaps vary by security; this approximates."""
    if price < 100:
        return 5
    elif price < 250:
        return 10
    elif price < 1000:
        return 20
    elif price < 3000:
        return 50
    else:
        return 100


def fetch_15m(tickers, period="60d"):
    """Batch download 15-min bars in chunks."""
    all_data = {}
    chunk_size = 50
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        try:
            df = yf.download(
                chunk, period=period, interval="15m",
                group_by="ticker", threads=True, progress=False,
                auto_adjust=False,
            )
        except Exception as e:
            print(f"  chunk {i}-{i+chunk_size} failed: {e}")
            continue
        for t in chunk:
            try:
                sub = df if len(chunk) == 1 else df[t]
                sub = sub.dropna(how="all")
                if not sub.empty:
                    all_data[t] = sub
            except Exception:
                continue
        time.sleep(1)
    return all_data


def build_universe():
    """NIFTY + BANKNIFTY + Nifty50/Next50/Midcap50 stocks (F&O proxy)."""
    indices = load_indices()
    stock_syms = list(dict.fromkeys(
        indices["nifty50"] + indices["nifty_next50"] + indices["midcap50"]
    ))
    instruments = dict(INDEX_INSTRUMENTS)
    for s in stock_syms:
        instruments[f"{s}.NS"] = (s, None, False)   # interval computed from price
    return instruments


def scan_ema_crossovers():
    instruments = build_universe()
    yf_tickers = list(instruments.keys())
    data = fetch_15m(yf_tickers)

    results = []
    scanned, errors = 0, 0

    for yf_tkr, df in data.items():
        try:
            name, fixed_interval, is_index = instruments[yf_tkr]
            df = df.dropna(subset=["Close"])
            if len(df) < 210:
                continue
            closes = df["Close"]

            ema100 = closes.ewm(span=100, adjust=False).mean()
            ema200 = closes.ewm(span=200, adjust=False).mean()

            scanned += 1

            curr_close, prev_close = float(closes.iloc[-1]), float(closes.iloc[-2])
            curr_e100, prev_e100   = float(ema100.iloc[-1]), float(ema100.iloc[-2])
            curr_e200, prev_e200   = float(ema200.iloc[-1]), float(ema200.iloc[-2])

            interval = fixed_interval if fixed_interval else strike_interval_for_price(curr_close)
            atm = round(curr_close / interval) * interval

            for label, curr_e, prev_e in (("EMA100", curr_e100, prev_e100),
                                           ("EMA200", curr_e200, prev_e200)):
                bull_cross = prev_close <= prev_e and curr_close > curr_e
                bear_cross = prev_close >= prev_e and curr_close < curr_e
                if not (bull_cross or bear_cross):
                    continue
                results.append({
                    "sym":       name,
                    "ema":       label,
                    "direction": "bullish" if bull_cross else "bearish",
                    "option":    "CE" if bull_cross else "PE",
                    "ltp":       round(curr_close, 2),
                    "atm":       int(atm) if atm == int(atm) else round(atm, 1),
                    "is_index":  is_index,
                })
        except Exception:
            errors += 1
            continue

    # Indices first, then stocks; bullish before bearish
    results.sort(key=lambda x: (not x["is_index"], x["direction"] != "bullish", x["sym"]))
    return results, scanned, errors


def format_message(results, scanned):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["<b>🎯 Options ATM Crossover Scan (15m EMA100/200)</b>", f"{now} · {scanned} scanned",
             "<i>Underlying crossed EMA100/200 on 15-min chart → ATM strike + direction (check live premium on your broker app)</i>"]
    if not results:
        lines.append("\nNo crossovers found.")
        return "\n".join(lines)

    lines.append(f"\n<b>{len(results)} crossover(s) found:</b>\n")
    for i, r in enumerate(results[:30], 1):
        arrow = "🔼" if r["direction"] == "bullish" else "🔽"
        tag = "📍" if r["is_index"] else ""
        lines.append(
            f"{i}. {tag}<b>{r['sym']}</b>  {arrow} {r['ema']} cross  "
            f"LTP {r['ltp']}  → <b>{r['atm']} {r['option']}</b>"
        )
    return "\n".join(lines)


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def main():
    print("Scanning NIFTY + BANKNIFTY + F&O stocks for 15m EMA100/200 crossovers...")
    results, scanned, errors = scan_ema_crossovers()
    print(f"Scanned: {scanned}  Errors: {errors}  Found: {len(results)}")
    msg = format_message(results, scanned)
    print("\n--- Telegram message ---")
    print(msg)
    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


if __name__ == "__main__":
    main()
