"""
Crypto Scanner → Telegram
============================
Tracks major cryptocurrencies (BTC, ETH, SOL, BNB, XRP, DOGE) using free
Yahoo Finance data. Crypto trades 24/7, so "today" here just means the
most recent UTC day of bars — same logic as the stock scanners, reused.

One combined message covering: RSI+EMA signal, day high/low proximity,
and trend R² (clean move vs sideways) for each coin.
"""
import json
import os
import time
import urllib.request
import urllib.parse

import yfinance as yf
import pandas as pd
import numpy as np

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID   = os.environ["TELEGRAM_CHAT_ID"]

COINS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "BNB-USD": "BNB",
    "XRP-USD": "XRP",
    "DOGE-USD": "Dogecoin",
}

SIGNAL_MIN_RSI    = 60
NEAR_EXTREME_PCT  = 0.5
MIN_R2            = 0.70


def ema(series, period):
    if len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def rsi(series, period=14):
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    last_gain, last_loss = avg_gain.iloc[-1], avg_loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def fetch_daily(tickers, period="1y"):
    df = yf.download(list(tickers), period=period, interval="1d",
                      group_by="ticker", threads=True, progress=False, auto_adjust=False)
    out = {}
    for t in tickers:
        try:
            sub = df if len(tickers) == 1 else df[t]
            sub = sub.dropna(how="all")
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue
    return out


def fetch_15m(tickers, period="5d"):
    df = yf.download(list(tickers), period=period, interval="15m",
                      group_by="ticker", threads=True, progress=False, auto_adjust=False)
    out = {}
    for t in tickers:
        try:
            sub = df if len(tickers) == 1 else df[t]
            sub = sub.dropna(how="all")
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue
    return out


def analyze():
    daily = fetch_daily(list(COINS.keys()))
    intraday = fetch_15m(list(COINS.keys()))
    rows = []

    for t, name in COINS.items():
        row = {"sym": name, "ticker": t}

        # Daily signal: RSI + EMA stack
        ddf = daily.get(t)
        if ddf is not None and len(ddf) >= 206:
            closes = ddf["Close"].dropna()
            ltp = float(closes.iloc[-1])
            e9, e20, e50, e100, e200 = (ema(closes, p) for p in (9, 20, 50, 100, 200))
            r = rsi(closes, 14)
            row["ltp"] = round(ltp, 2)
            row["rsi"] = round(r, 1) if r is not None else None
            if None not in (e9, e20, e50, e100, e200):
                row["ema_bullish"] = ltp > e9 > e20 > e50 > e100 > e200
                row["ema_bearish"] = ltp < e9 < e20 < e50 < e100 < e200
            else:
                row["ema_bullish"] = row["ema_bearish"] = None
            row["signal"] = bool(row["ema_bullish"] and r is not None and r >= SIGNAL_MIN_RSI)

        # Intraday: day high/low proximity + trend R^2
        idf = intraday.get(t)
        if idf is not None and not idf.empty:
            idf = idf.dropna(subset=["Close", "High", "Low"])
            today_date = idf.index[-1].date()
            today_bars = idf[idf.index.date == today_date]
            if len(today_bars) >= 5:
                day_high = float(today_bars["High"].max())
                day_low = float(today_bars["Low"].min())
                ltp_i = float(today_bars["Close"].iloc[-1])
                dist_high = (day_high - ltp_i) / day_high * 100 if day_high > 0 else 999
                dist_low = (ltp_i - day_low) / day_low * 100 if day_low > 0 else 999
                row["near_high"] = dist_high <= NEAR_EXTREME_PCT
                row["near_low"] = dist_low <= NEAR_EXTREME_PCT

                closes_i = today_bars["Close"].values.astype(float)
                x = np.arange(len(closes_i))
                slope, intercept = np.polyfit(x, closes_i, 1)
                pred = slope * x + intercept
                ss_res = float(((closes_i - pred) ** 2).sum())
                ss_tot = float(((closes_i - closes_i.mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                row["r2"] = round(r2, 2)
                row["trend_dir"] = "up" if slope > 0 else "down"
                row["clean_trend"] = r2 >= MIN_R2

        rows.append(row)

    return rows


def format_message(rows):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = ["<b>₿ Crypto Scan — BTC/ETH/SOL/BNB/XRP/DOGE</b>", f"{now}", ""]

    for r in rows:
        if "ltp" not in r:
            lines.append(f"<b>{r['sym']}</b>: no data")
            continue
        tags = []
        if r.get("signal"):
            tags.append("✅ SIGNAL (RSI&gt;=60 + EMA bullish)")
        elif r.get("ema_bullish"):
            tags.append("🔼 EMA bullish stack")
        elif r.get("ema_bearish"):
            tags.append("🔽 EMA bearish stack")
        if r.get("near_high"):
            tags.append("at day high")
        if r.get("near_low"):
            tags.append("at day low")
        if r.get("clean_trend"):
            arrow = "🔼" if r.get("trend_dir") == "up" else "🔽"
            tags.append(f"{arrow} clean trend R&#178;={r.get('r2')}")
        tag_str = " · ".join(tags) if tags else "no signal"

        lines.append(f"<b>{r['sym']}</b>  ${r['ltp']:,}  RSI {r.get('rsi')}")
        lines.append(f"  {tag_str}\n")

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
    print("Analyzing crypto...")
    rows = analyze()
    for r in rows:
        print(" ", r)
    msg = format_message(rows)
    print("\n--- Telegram message ---")
    print(msg)
    result = send_telegram(msg)
    print("\nTelegram send result:", result.get("ok"))


if __name__ == "__main__":
    main()
