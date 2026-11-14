"""Rule-based setups: the rules propose candidate trades, the model decides which to take.

Each setup fires on the candle where its condition becomes true (and was absent for
`cooldown` candles), so events are sparse and overlap far less than per-candle samples.
"""
import numpy as np

from .features import build_features, feature_columns

SETUPS = ("breakout", "reversal", "pullback")
# Features whose sign means "up" vs "down"; they are mirrored for short candidates so that
# "with the trend" looks the same to the model for longs and shorts.
DIRECTIONAL = ("rsi", "di_diff", "ema_dist", "ema_cross", "macd", "bb", "ret1", "ret3", "body")


def find_setups(frame, lookback=48, cooldown=12):
    """Returns a list of (row, setup_index, side) sorted by row."""
    c = frame["close"]
    chan_hi = frame["high"].rolling(lookback).max().shift(1)
    chan_lo = frame["low"].rolling(lookback).min().shift(1)
    adx1h, di1h = frame["f_1h_adx"], frame["f_1h_di_diff"]
    rsi15 = frame["f_15m_rsi"]
    up4h = (frame["f_4h_ema_cross"] > 0) & (frame["f_4h_di_diff"] > 0)
    dn4h = (frame["f_4h_ema_cross"] < 0) & (frame["f_4h_di_diff"] < 0)
    conds = [
        # breakout: price leaves its 4-hour channel in the direction of a trending 1h chart
        (0, 1, (c > chan_hi) & (adx1h > 0.20) & (di1h > 0)),
        (0, -1, (c < chan_lo) & (adx1h > 0.20) & (di1h < 0)),
        # reversal: 15m RSI extreme while the 1h chart is not trending
        (1, 1, (rsi15 < -0.20) & (adx1h < 0.20)),
        (1, -1, (rsi15 > 0.20) & (adx1h < 0.20)),
        # pullback: dip (15m RSI) against a 4h trend
        (2, 1, up4h & (rsi15 < -0.15)),
        (2, -1, dn4h & (rsi15 > 0.15)),
    ]
    out = []
    for setup, side, cond in conds:
        cond = cond.fillna(False).astype(float)
        fresh = (cond > 0) & ~(cond.shift(1).rolling(cooldown).max() > 0)
        out += [(int(r), setup, side) for r in np.nonzero(fresh.to_numpy())[0]]
    return sorted(out)


def event_feature_names(cols):
    return list(cols) + [f"setup_{s}" for s in SETUPS] + ["side"]


def prepare_symbol(candles, cfg):
    """Candles of one symbol -> (bars, events).

    bars:   array [close_ms, high, low, close, atr] per base candle
    events: {row: [(setup_index, side, feature_vector), ...]}
    """
    frame = build_features(candles, cfg.base_tf, cfg.atr_tf)
    cols = feature_columns(frame)
    mirror = np.array([any(c.endswith("_" + d) for d in DIRECTIONAL) for c in cols])
    X = frame[cols].to_numpy(np.float32)
    events = {}
    for row, setup, side in find_setups(frame, cfg.breakout_lookback, cfg.setup_cooldown):
        x = X[row] * np.where(mirror, side, 1).astype(np.float32)
        onehot = np.zeros(len(SETUPS) + 1, np.float32)
        onehot[setup], onehot[-1] = 1, side
        events.setdefault(row, []).append((setup, side, np.concatenate([x, onehot])))
    bars = frame[["close_ms", "high", "low", "close", "atr_ref"]].to_numpy(float)
    return bars, events
