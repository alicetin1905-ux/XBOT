import numpy as np
import pandas as pd

from xbot import indicators as ta
from xbot.broker import PaperBroker
from xbot.config import Config
from xbot.data import TF_MS
from xbot.engine import Engine
from xbot.features import build_features, feature_columns
from xbot.model import OnlineLogit


def make_candles(tf, n_base=6000, seed=1):
    """Random-walk 5m candles, aggregated to `tf`."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n_base)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open_ms": np.arange(n_base) * 300_000, "open": open_,
                       "high": np.maximum(open_, close) * 1.001,
                       "low": np.minimum(open_, close) * 0.999,
                       "close": close, "volume": rng.uniform(1, 10, n_base)})
    k = TF_MS[tf] // 300_000
    g = df.groupby(df.index // k).agg(open_ms=("open_ms", "first"), open=("open", "first"),
                                      high=("high", "max"), low=("low", "min"),
                                      close=("close", "last"), volume=("volume", "sum"))
    g["close_ms"] = g["open_ms"] + TF_MS[tf]
    return g.reset_index(drop=True)


def all_candles(n_base=6000):
    return {tf: make_candles(tf, n_base) for tf in Config().timeframes}


def test_indicator_ranges():
    df = make_candles("5m")
    r = ta.rsi(df["close"]).dropna()
    a, pdi, mdi = ta.adx(df["high"], df["low"], df["close"])
    assert r.between(0, 100).all()
    assert a.dropna().between(0, 100).all() and (pdi.dropna() >= 0).all()
    up = pd.Series(np.arange(1.0, 60))
    assert ta.rsi(up).iloc[-1] == 100


def test_features_have_no_lookahead():
    full = build_features(all_candles())
    cut_ms = full["close_ms"].iloc[len(full) // 2]
    truncated = {tf: df[df["close_ms"] <= cut_ms] for tf, df in all_candles().items()}
    part = build_features(truncated)
    cols = feature_columns(full)
    a = full[full["close_ms"] == cut_ms][cols].to_numpy()
    b = part[part["close_ms"] == cut_ms][cols].to_numpy()
    assert len(cols) == 13 * 5 + 2
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_model_learns_separable_problem():
    rng = np.random.default_rng(0)
    m = OnlineLogit(3, lr=0.05, l2=0.0)
    for _ in range(4000):
        x = rng.normal(size=3)
        m.observe(x)
        m.update(x, float(x[0] - x[1] > 0))
    X = rng.normal(size=(500, 3))
    acc = np.mean([(m.predict(x) > 0.5) == (x[0] - x[1] > 0) for x in X])
    assert acc > 0.95
    m2 = OnlineLogit.from_dict(m.to_dict())
    assert m2.predict(X[0]) == m.predict(X[0])


def test_broker_long_target_and_short_stop():
    b = PaperBroker(1000, fee=0.001, slippage=0)
    b.open(1, 2, 100, sl=95, tp=110, bars=10, time_ms=0)
    assert b.on_bar(105, 99, 104, 1) is None
    t = b.on_bar(111, 104, 110, 2)
    assert t["reason"] == "target"
    assert abs(t["pnl"] - (20 - 0.2 - 0.22)) < 1e-9 and abs(b.balance - 1019.58) < 1e-9
    b.open(-1, 1, 100, sl=105, tp=90, bars=10, time_ms=3)
    t = b.on_bar(106, 89, 100, 4)  # both hit in one candle -> stop assumed
    assert t["reason"] == "stop" and t["pnl"] < -5


def test_broker_timeout():
    b = PaperBroker(1000, fee=0, slippage=0)
    b.open(1, 1, 100, sl=90, tp=110, bars=2, time_ms=0)
    assert b.on_bar(101, 99, 100, 1) is None
    assert b.on_bar(102, 99, 101, 2)["reason"] == "timeout"


def test_engine_runs_and_roundtrips_state(tmp_path):
    frame = build_features(all_candles())
    cfg = Config(state_dir=str(tmp_path), min_train=200, horizon=24)
    eng = Engine(cfg, len(feature_columns(frame)))
    half = len(frame) // 2
    eng.run_frame(frame.iloc[:half])
    assert eng.model.updates > 0 and len(eng.pending) <= cfg.horizon
    eng.save()
    eng2 = Engine.load(cfg)
    assert eng2.last_ms == eng.last_ms and eng2.model.updates == eng.model.updates
    r1 = eng.run_frame(frame)
    r2 = eng2.run_frame(frame)
    assert len(r1) == len(frame) - half
    assert [r["p"] for r in r1] == [r["p"] for r in r2]
    assert eng.broker.balance == eng2.broker.balance
    assert eng.run_frame(frame) == []  # nothing is processed twice
