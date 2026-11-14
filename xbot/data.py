"""Candle download from the public Binance market-data API (no API key needed)."""
import os
import time

import pandas as pd
import requests

BASE_URL = os.environ.get("XBOT_API", "https://data-api.binance.vision")
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000}
COLUMNS = ["open_ms", "open", "high", "low", "close", "volume"]


def _get(params, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_klines(symbol, tf, start_ms=None, end_ms=None, limit=1000):
    """Closed candles for one timeframe, oldest first, with a `close_ms` column.

    Without start_ms the most recent `limit` candles are returned.
    """
    step = TF_MS[tf]
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms or now_ms, now_ms)
    rows = []
    if start_ms is None:
        rows = _get({"symbol": symbol, "interval": tf, "limit": limit})
    else:
        cursor = start_ms
        while cursor < end_ms:
            batch = _get({"symbol": symbol, "interval": tf, "startTime": cursor, "limit": 1000})
            if not batch:
                break
            rows.extend(batch)
            cursor = batch[-1][0] + step
            if len(batch) < 1000:
                break
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS).astype(float)
    df["open_ms"] = df["open_ms"].astype("int64")
    df["close_ms"] = df["open_ms"] + step
    df = df[df["close_ms"] <= end_ms]  # drop the candle that is still forming
    return df.drop_duplicates("open_ms").sort_values("open_ms").reset_index(drop=True)


def fetch_all(symbol, timeframes, days=None, limit=300):
    """Candles for every timeframe. `days` of history, or the last `limit` candles."""
    out = {}
    now_ms = int(time.time() * 1000)
    for tf in timeframes:
        if days is None:
            out[tf] = fetch_klines(symbol, tf, limit=limit)
        else:
            warmup = 150 * TF_MS[tf]  # extra candles so indicators are settled at the start
            out[tf] = fetch_klines(symbol, tf, start_ms=now_ms - int(days * 86_400_000) - warmup)
    return out
