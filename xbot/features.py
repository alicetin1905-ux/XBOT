"""Turn candles of several timeframes into one feature row per base candle.

Every feature is scale-free (ratios / ATR-normalised) so the same model works at
any price level. Higher timeframes are joined on candle *close* time, so a row
only ever contains candles that were already closed at that moment (no lookahead).
"""
import numpy as np
import pandas as pd

from . import indicators as ta


def tf_features(df, tf):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    atr = ta.atr(h, l, c).replace(0, np.nan)
    adx, pdi, mdi = ta.adx(h, l, c)
    ema20, ema50 = ta.ema(c, 20), ta.ema(c, 50)
    vol_ma = v.rolling(20).mean().replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    p = f"f_{tf}_"
    out = pd.DataFrame({
        "close_ms": df["close_ms"],
        p + "rsi": ta.rsi(c) / 100 - 0.5,
        p + "adx": adx / 100,
        p + "di_diff": (pdi - mdi) / 100,
        p + "vol_ratio": np.log((v / vol_ma).clip(lower=1e-3)),
        p + "vol_trend": np.log((v.rolling(5).mean() / vol_ma).clip(lower=1e-3)),
        p + "ema_dist": (c - ema50) / atr,
        p + "ema_cross": (ema20 - ema50) / atr,
        p + "macd": ta.macd_hist(c) / atr,
        p + "bb": ta.bollinger_pct_b(c) - 0.5,
        p + "atr_pct": np.log(atr / c),
        p + "ret1": c.diff() / atr,
        p + "ret3": c.diff(3) / atr,
        p + "body": ((c - o) / rng).fillna(0),
        f"atr_{tf}": atr,
    })
    return out


def build_features(candles, base_tf="5m", atr_tf="1h"):
    """candles: {tf: DataFrame from data.fetch_klines}. Returns one row per base candle."""
    base = candles[base_tf]
    frame = base[["close_ms", "open", "high", "low", "close", "volume"]].copy()
    for tf, df in candles.items():
        frame = pd.merge_asof(frame, tf_features(df, tf), on="close_ms", direction="backward")
    hour = (frame["close_ms"] / 3_600_000.0) % 24
    frame["f_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["f_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["atr_ref"] = frame[f"atr_{atr_tf}"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame


def feature_columns(frame):
    return [c for c in frame.columns if c.startswith("f_")]
