"""Technical indicators on pandas Series (Wilder smoothing where customary)."""
import numpy as np
import pandas as pd


def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close, n=14):
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder((-d).clip(lower=0), n)
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan)).fillna(np.inf)


def true_range(high, low, close):
    pc = close.shift()
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n=14):
    return _wilder(true_range(high, low, close), n)


def adx(high, low, close, n=14):
    """Returns (adx, +DI, -DI)."""
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = _wilder(true_range(high, low, close), n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / tr
    minus_di = 100 * _wilder(minus_dm, n) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx.fillna(0), n), plus_di, minus_di


def macd_hist(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    return line - line.ewm(span=signal, adjust=False, min_periods=signal).mean()


def bollinger_pct_b(close, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0).replace(0, np.nan)
    return (close - (mid - k * sd)) / (2 * k * sd)
