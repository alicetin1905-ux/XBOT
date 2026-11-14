"""Candle download from the public Binance market-data API (no API key needed)."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
    else:   # download 1000-candle chunks in parallel
        chunk = 1000 * step
        starts = range(int(start_ms), int(end_ms), chunk)
        with ThreadPoolExecutor(8) as pool:
            for batch in pool.map(lambda t: _get({"symbol": symbol, "interval": tf, "limit": 1000,
                                                  "startTime": t, "endTime": t + chunk - 1}), starts):
                rows.extend(batch)
    df = pd.DataFrame([r[:6] for r in rows], columns=COLUMNS).astype(float)
    df["open_ms"] = df["open_ms"].astype("int64")
    df["close_ms"] = df["open_ms"] + step
    df = df[df["close_ms"] <= end_ms]  # drop the candle that is still forming
    return df.drop_duplicates("open_ms").sort_values("open_ms").reset_index(drop=True)


def load_candles(symbol, tf, days, cache_dir="data"):
    """`days` of closed candles (plus indicator warm-up), using and extending a local CSV cache."""
    os.makedirs(cache_dir, exist_ok=True)
    step = TF_MS[tf]
    need_start = int(time.time() * 1000) - int(days * 86_400_000) - 150 * step
    path = os.path.join(cache_dir, f"{symbol}_{tf}.csv")
    meta_path = os.path.join(cache_dir, "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    key = f"{symbol}_{tf}"
    cached = None
    if os.path.exists(path) and meta.get(key, float("inf")) <= need_start:
        cached = pd.read_csv(path)
    start = int(cached["open_ms"].iloc[-1]) if cached is not None and len(cached) else need_start
    new = fetch_klines(symbol, tf, start_ms=start)
    df = new if cached is None else pd.concat([cached, new])
    df = df.drop_duplicates("open_ms", keep="last").sort_values("open_ms").reset_index(drop=True)
    df.to_csv(path, index=False)
    meta[key] = min(meta.get(key, need_start), need_start) if cached is not None else need_start
    json.dump(meta, open(meta_path, "w"))
    return df[df["open_ms"] >= need_start].reset_index(drop=True)


def fetch_recent(symbol, timeframes, limit=300):
    """The last `limit` closed candles of every timeframe (for the live loop)."""
    return {tf: fetch_klines(symbol, tf, limit=limit) for tf in timeframes}


def load_history(symbol, timeframes, days, cache_dir="data"):
    return {tf: load_candles(symbol, tf, days, cache_dir) for tf in timeframes}
