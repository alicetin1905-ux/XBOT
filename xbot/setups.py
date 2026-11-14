"""Rule-based setups: the rules propose candidate trades, the model decides which to take.

Each setup fires on the candle where its condition becomes true (and was absent for
`cooldown` candles), so events are sparse and overlap far less than per-candle samples.
"""
import numpy as np

from .features import build_features, feature_columns

SETUPS = ("breakout", "reversal", "pullback", "fakeout")
# Features whose sign means "up" vs "down"; they are mirrored for short candidates so that
# "with the trend" looks the same to the model for longs and shorts.
DIRECTIONAL = ("rsi", "di_diff", "ema_dist", "ema_cross", "macd", "bb", "ret1", "ret3", "body",
               "buy", "buy_ma", "fund", "fund_avg")


def trend(frame):
    """Trend per base candle from the 30m and 1h charts: +1 up, -1 down, 0 sideways.

    Up means that on both charts EMA20 is above EMA50 and +DI is above -DI.
    """
    up = ((frame["f_30m_ema_cross"] > 0) & (frame["f_30m_di_diff"] > 0)
          & (frame["f_1h_ema_cross"] > 0) & (frame["f_1h_di_diff"] > 0))
    dn = ((frame["f_30m_ema_cross"] < 0) & (frame["f_30m_di_diff"] < 0)
          & (frame["f_1h_ema_cross"] < 0) & (frame["f_1h_di_diff"] < 0))
    return up.astype(int) - dn.astype(int)


def find_setups(frame, lookback=24, cooldown=12, fake_lookback=12, fake_depth=1.5):
    """Returns a list of (row, setup_index, side) sorted by row.

    Entries come from the 5m and 15m charts and only go with the 30m/1h trend;
    without a trend only the reversal setup is allowed.
    fake_lookback: candles that define the level for a fakeout; fake_depth: how far price
    may go through that level, in 5m ATRs, to still count as a *small* false break.
    """
    c = frame["close"]
    chan_hi = frame["high"].rolling(lookback).max().shift(1)
    chan_lo = frame["low"].rolling(lookback).min().shift(1)
    rsi5, rsi15 = frame["f_5m_rsi"], frame["f_15m_rsi"]
    tr = trend(frame)
    up, dn, flat = tr > 0, tr < 0, tr == 0
    # fakeout: this or the previous 5m candle pokes through the recent low (in an uptrend)
    # by a small amount, and this candle closes back above that level as a rising candle
    o, h, l, atr5 = frame["open"], frame["high"], frame["low"], frame["atr_5m"]
    lvl_lo = l.rolling(fake_lookback).min().shift(2)
    lvl_hi = h.rolling(fake_lookback).max().shift(2)
    dip = lvl_lo - np.minimum(l, l.shift(1))        # how far price went below the level
    spike = np.maximum(h, h.shift(1)) - lvl_hi
    fake_long = up & (dip > 0) & (dip < fake_depth * atr5) & (c > lvl_lo) & (c > o)
    fake_short = dn & (spike > 0) & (spike < fake_depth * atr5) & (c < lvl_hi) & (c < o)
    conds = [
        (3, 1, fake_long),
        (3, -1, fake_short),
        # breakout: 5m close leaves its recent channel in the direction of the trend
        (0, 1, (c > chan_hi) & up),
        (0, -1, (c < chan_lo) & dn),
        # reversal: 15m RSI extreme while there is no trend
        (1, 1, (rsi15 < -0.20) & flat),
        (1, -1, (rsi15 > 0.20) & flat),
        # pullback: 5m or 15m RSI dips against the trend
        (2, 1, up & ((rsi5 < -0.20) | (rsi15 < -0.15))),
        (2, -1, dn & ((rsi5 > 0.20) | (rsi15 > 0.15))),
    ]
    out = []
    for setup, side, cond in conds:
        cond = cond.fillna(False).astype(float)
        fresh = (cond > 0) & ~(cond.shift(1).rolling(cooldown).max() > 0)
        out += [(int(r), setup, side) for r in np.nonzero(fresh.to_numpy())[0]]
    return sorted(out)


def event_feature_names(cols):
    return list(cols) + [f"setup_{s}" for s in SETUPS] + ["side"]


def prepare_symbol(candles, cfg, derivs=None):
    """Candles of one symbol -> (bars, events).

    bars:   array [close_ms, high, low, close, atr, trend] per base candle
    events: {row: [(setup_index, side, feature_vector), ...]}
    """
    frame = build_features(candles, cfg.base_tf, cfg.atr_tf, derivs)
    cols = feature_columns(frame)
    mirror = np.array([any(c.endswith("_" + d) for d in DIRECTIONAL) for c in cols])
    X = frame[cols].to_numpy(np.float32)
    events = {}
    for row, setup, side in find_setups(frame, cfg.breakout_lookback, cfg.setup_cooldown,
                                         cfg.fakeout_lookback, cfg.fakeout_depth_atr):
        x = X[row] * np.where(mirror, side, 1).astype(np.float32)
        onehot = np.zeros(len(SETUPS) + 1, np.float32)
        onehot[setup], onehot[-1] = 1, side
        events.setdefault(row, []).append((setup, side, np.concatenate([x, onehot])))
    bars = np.column_stack([frame[["close_ms", "high", "low", "close", "atr_ref"]].to_numpy(float),
                            trend(frame).to_numpy(float)])
    return bars, events
