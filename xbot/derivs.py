"""Futures data: open interest and funding rate.

History comes from Binance's public bulk files (data.binance.vision). Live data comes from
the Binance futures API or, where that is not reachable, from OKX. Open-interest levels
differ per exchange, so features are only ever computed *within* one source (as relative
changes) and then combined.
"""
import io
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

VISION = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com"
OKX = "https://www.okx.com"
DAY = 86_400_000
DERIV_FEATURES = ("f_oi_chg_1h", "f_oi_chg_4h", "f_oi_chg_24h", "f_fund", "f_fund_avg")
_live_oi = {}       # symbol -> accumulated live open-interest series
_live_source = {}   # remembers which live source works


def _json(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)


def _zip_csv(url, retries=4):
    """A zipped CSV from the bulk archive, or None if that file does not exist (yet)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return pd.read_csv(z.open(z.namelist()[0]))
        except (requests.RequestException, zipfile.BadZipFile):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _hour(ms):
    """Funding timestamps rounded to the hour (sources differ by a few milliseconds)."""
    return (np.round(np.asarray(ms, float) / 3_600_000) * 3_600_000).astype(np.int64)


def _merge(old, new):
    df = new if old is None else pd.concat([old, new])
    return df.dropna().drop_duplicates("ms", keep="last").sort_values("ms").reset_index(drop=True)


# ---- history (bulk files, cached) ----------------------------------------
def _oi_day(symbol, day):
    df = _zip_csv(f"{VISION}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip")
    if df is None:
        return None
    ms = (pd.to_datetime(df["create_time"], utc=True) - pd.Timestamp(0, tz="UTC")) \
        // pd.Timedelta(milliseconds=1)
    return pd.DataFrame({"ms": ms.astype("int64"), "oi": df["sum_open_interest"].astype(float)})


def _funding_month(symbol, month):
    df = _zip_csv(f"{VISION}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip")
    if df is None:
        return None
    return pd.DataFrame({"ms": _hour(df["calc_time"]),
                         "rate": df["last_funding_rate"].astype(float)})


def load_derivs_history(symbol, days, cache_dir="data"):
    """{"oi": [history, live], "fund": series} covering `days`, using a local cache."""
    os.makedirs(cache_dir, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=days + 2)
    out = {}
    for kind, freq, fmt, fetch in (("oi", "D", "%Y-%m-%d", _oi_day),
                                   ("funding", "MS", "%Y-%m", _funding_month)):
        path = os.path.join(cache_dir, f"{symbol}_{kind}.csv")
        cached = pd.read_csv(path) if os.path.exists(path) else None
        have = set() if cached is None else set(
            pd.to_datetime(cached["ms"], unit="ms").dt.strftime(fmt))
        first = start.normalize() if freq == "D" else start.normalize().replace(day=1)
        wanted = [d.strftime(fmt) for d in pd.date_range(first, now, freq=freq)]
        todo = [d for d in wanted if d not in have or d == wanted[-1]]
        with ThreadPoolExecutor(16) as pool:
            parts = [p for p in pool.map(lambda d: fetch(symbol, d), todo) if p is not None]
        if parts:
            cached = _merge(cached, pd.concat(parts))
            cached.to_csv(path, index=False)
        out[kind] = cached if cached is not None else pd.DataFrame({"ms": [], "oi": []})
    result = {"oi": [out["oi"]], "fund": out["funding"]}
    try:    # the bulk files lag a day or more: top up with live data
        live = fetch_derivs_recent(symbol)
        result["oi"].append(live["oi"][0])
        result["fund"] = _merge(result["fund"], live["fund"])
    except Exception as exc:
        print(f"  ({symbol}: geen live futures-data voor de laatste dag: {exc!r})")
    return result


# ---- live ------------------------------------------------------------------
def _binance_live(symbol):
    oi = _json(f"{FAPI}/futures/data/openInterestHist",
               {"symbol": symbol, "period": "5m", "limit": 500})
    fund = _json(f"{FAPI}/fapi/v1/fundingRate", {"symbol": symbol, "limit": 100})
    return (pd.DataFrame({"ms": [int(r["timestamp"]) for r in oi],
                          "oi": [float(r["sumOpenInterest"]) for r in oi]}),
            pd.DataFrame({"ms": _hour([r["fundingTime"] for r in fund]),
                          "rate": [float(r["fundingRate"]) for r in fund]}))


def _okx_live(symbol, pages):
    inst = f"{symbol[:-4]}-USDT-SWAP"
    rows, end = [], None
    for _ in range(pages):
        params = {"instId": inst, "period": "5m", "limit": 100}
        if end:
            params["end"] = end
        data = _json(f"{OKX}/api/v5/rubik/stat/contracts/open-interest-history", params)["data"]
        if not data:
            break
        rows += data
        end = min(int(r[0]) for r in data)
    fund = _json(f"{OKX}/api/v5/public/funding-rate-history", {"instId": inst, "limit": 100})["data"]
    return (pd.DataFrame({"ms": [int(r[0]) for r in rows], "oi": [float(r[1]) for r in rows]}),
            pd.DataFrame({"ms": _hour([r["fundingTime"] for r in fund]),
                          "rate": [float(r["realizedRate"] or r["fundingRate"]) for r in fund]}))


def fetch_derivs_recent(symbol):
    """Recent open interest (>= 24h, 5-minute steps) and funding rates from a live API."""
    known = _live_oi.get(symbol)
    if _live_source.get("name") != "okx":
        try:
            oi, fund = _binance_live(symbol)
            _live_source["name"] = "binance"
        except requests.RequestException:
            if _live_source.get("name") == "binance":
                raise
            _live_source["name"] = "okx"
    if _live_source["name"] == "okx":
        oi, fund = _okx_live(symbol, pages=1 if known is not None else 4)
    oi = _merge(known, oi)
    oi = oi[oi["ms"] >= oi["ms"].iloc[-1] - 3 * DAY].reset_index(drop=True)
    _live_oi[symbol] = oi
    return {"oi": [oi], "fund": _merge(None, fund)}


# ---- features --------------------------------------------------------------
def _asof(query_ms, src_ms, src_val, tolerance):
    """Latest source value at or before each query time, NaN if older than `tolerance`."""
    if not len(src_ms):
        return np.full(len(query_ms), np.nan)
    i = np.searchsorted(src_ms, query_ms, side="right") - 1
    ok = (i >= 0) & (query_ms - src_ms[np.clip(i, 0, None)] <= tolerance)
    return np.where(ok, src_val[np.clip(i, 0, None)], np.nan)


def deriv_features(close_ms, derivs=None):
    """Futures features per base candle. Missing data gives NaN (the model handles that)."""
    close_ms = np.asarray(close_ms, np.int64)
    out = pd.DataFrame({c: np.nan for c in DERIV_FEATURES}, index=range(len(close_ms)))
    if not derivs:
        return out
    t = close_ms - 1    # strictly before the candle close: a snapshot stamped at the close
    tol = 15 * 60_000   # may be published seconds later
    for oi in derivs["oi"]:
        ms, val = oi["ms"].to_numpy(np.int64), oi["oi"].to_numpy(float)
        cur = _asof(t, ms, val, tol)
        for name, lag in (("f_oi_chg_1h", 1), ("f_oi_chg_4h", 4), ("f_oi_chg_24h", 24)):
            with np.errstate(divide="ignore", invalid="ignore"):
                chg = np.log(cur / _asof(t - lag * 3_600_000, ms, val, tol))
            chg[~np.isfinite(chg)] = np.nan
            out[name] = out[name].fillna(pd.Series(chg))
    f = derivs["fund"]
    if len(f):
        ms = f["ms"].to_numpy(np.int64)
        out["f_fund"] = _asof(t, ms, f["rate"].to_numpy(float) * 1e4, 12 * 3_600_000)
        out["f_fund_avg"] = _asof(t, ms, f["rate"].rolling(9, min_periods=3).mean().to_numpy(float) * 1e4,
                                  12 * 3_600_000)
    return out
