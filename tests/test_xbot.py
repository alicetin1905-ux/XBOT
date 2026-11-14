import numpy as np
import pandas as pd

from xbot import indicators as ta
from xbot.config import Config
from xbot.data import TF_MS
from xbot.derivs import deriv_features
from xbot.engine import Engine
from xbot.features import build_features, feature_columns
from xbot.setups import find_setups, prepare_symbol, trend


def make_candles(tf, n_base=6000, seed=1):
    """Random-walk 5m candles, aggregated to `tf`."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n_base)))
    open_ = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open_ms": np.arange(n_base) * 300_000, "open": open_,
                       "high": np.maximum(open_, close) * 1.001,
                       "low": np.minimum(open_, close) * 0.999,
                       "close": close, "volume": rng.uniform(1, 10, n_base)})
    df["taker_buy"] = df["volume"] * rng.uniform(0.3, 0.7, n_base)
    k = TF_MS[tf] // 300_000
    g = df.groupby(df.index // k).agg(open_ms=("open_ms", "first"), open=("open", "first"),
                                      high=("high", "max"), low=("low", "min"),
                                      close=("close", "last"), volume=("volume", "sum"),
                                      taker_buy=("taker_buy", "sum"))
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
    assert len(cols) == 15 * 5 + 2 + 5
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def prepared(n_base=6000, symbols=("AAA", "BBB")):
    cfg = Config(symbols=symbols, min_train=30, horizon=24, retrain_hours=48)
    out = {}
    for k, sym in enumerate(symbols):
        candles = {tf: make_candles(tf, n_base, seed=k + 1) for tf in cfg.timeframes}
        out[sym] = prepare_symbol(candles, cfg)
    return cfg, out


def test_setups_are_sparse_and_mirrored():
    frame = build_features(all_candles())
    events = find_setups(frame, 24, 12)
    assert 20 < len(events) < len(frame) / 10
    assert {s for _, s, _ in events} <= {0, 1, 2} and {d for _, _, d in events} == {1, -1}
    tr = trend(frame).to_numpy()
    assert set(tr) == {-1, 0, 1}
    for row, setup, side in events:     # trend setups only go with the trend
        assert tr[row] == (0 if setup == 1 else side)
    cfg, prep = prepared()
    bars, ev = prep["AAA"]
    assert (bars[:, 5] == tr).all()
    cols = feature_columns(frame)
    j = cols.index("f_1h_rsi")
    for row, lst in ev.items():
        for setup, side, x in lst:
            assert len(x) == len(cols) + 4 and x[-1] == side and x[len(cols) + setup] == 1
            assert np.isclose(x[j], side * build_row(frame, bars, row)[j], atol=1e-6)


def build_row(frame, bars, row):
    assert frame["close_ms"].iloc[row] == bars[row, 0]
    return frame[feature_columns(frame)].iloc[row].to_numpy()


def test_candidate_outcomes_and_trade_accounting():
    cfg = Config(order_type="market", fee=0.001, slippage=0, min_train=10**9,
                 gate_min_samples=0, gate_z=-1e9)
    eng = Engine(cfg)
    eng.model.predict = lambda x: 1.0          # always expect profit
    eng.gate.extend([1.0])
    x = np.zeros(3, np.float32)
    # long at 100, ATR 2.5 -> barrier 5% -> sl 95, tp 105
    out = eng.on_bar("AAA", 300_000, 101, 99, 100, 2.5, events=[(0, 1, x)])
    pos = out["opened"][0]
    assert pos["sl"] == 95 and abs(pos["tp"] - 105) < 1e-9
    qty = pos["qty"]
    assert abs(qty - 10_000 * 0.01 / (0.05 * 100)) < 1e-9   # stop costs 1% of equity
    assert eng.on_bar("AAA", 600_000, 104, 99, 103, 2.5)["closed"] == []
    t = eng.on_bar("AAA", 900_000, 106, 103, 105, 2.5)["closed"][0]
    assert t["reason"] == "target"
    assert abs(t["pnl"] - (qty * 5 - qty * 100 * 0.001 - qty * 105 * 0.001)) < 1e-9
    assert abs(eng.balance - (10_000 + t["pnl"])) < 1e-9
    assert abs(eng.r[0] - (0.05 - 0.002) / 0.05) < 1e-9
    # short where stop and target fall in the same candle -> counted as a stop
    eng.on_bar("AAA", 1_200_000, 101, 99, 100, 2.5, events=[(1, -1, x)])
    t = eng.on_bar("AAA", 1_500_000, 106, 94, 100, 2.5)["closed"][0]
    assert t["reason"] == "stop" and t["pnl"] < 0 and eng.r[1] < -1


def test_no_trade_without_model_or_with_closed_gate():
    eng = Engine(Config())
    x = np.zeros(3, np.float32)
    assert eng.on_bar("AAA", 300_000, 101, 99, 100, 2.5, events=[(0, 1, x)])["opened"] == []
    eng.model.predict = lambda x: 1.0
    assert eng.on_bar("AAA", 600_000, 101, 99, 100, 2.5, events=[(0, 1, x)])["opened"] == []
    assert len(eng.open["AAA"]) == 2          # still followed as candidates to learn from


def test_engine_learns_runs_and_roundtrips_state(tmp_path):
    cfg, prep = prepared()
    cfg.state_dir = str(tmp_path)
    eng = Engine(cfg)
    mid = int(prep["AAA"][0][len(prep["AAA"][0]) // 2, 0])
    eng.run(prep, until_ms=mid)
    assert eng.model.reg is not None and len(eng.r) > cfg.min_train
    assert all(ms < mid for ms in eng.last_ms.values())
    eng.save()
    eng2 = Engine.load(cfg)
    assert eng2.last_ms == eng.last_ms and len(eng2.r) == len(eng.r)
    eng.run(prep)
    eng2.run(prep)
    assert len(eng.log) > 0
    assert [l[4] for l in eng.log[-200:]] == [l[4] for l in eng2.log[-200:]]
    assert eng.balance == eng2.balance and len(eng.r) == len(eng2.r)
    assert eng.run(prep) == []                # nothing is processed twice


def test_training_only_uses_resolved_candidates():
    cfg, prep = prepared()
    eng = Engine(cfg)
    seen = []
    orig = eng.retrain
    def spy(now):
        ok = orig(now)
        if ok:
            seen.append((now, max(eng.t)))
        return ok
    eng.retrain = spy
    eng.run(prep)
    assert seen and all(t_event < now for now, t_event in seen)


def test_deriv_features_use_only_past_data():
    oi = pd.DataFrame({"ms": np.arange(0, 400) * 300_000, "oi": 1000.0 + np.arange(400)})
    fund = pd.DataFrame({"ms": [0, 8 * 3_600_000], "rate": [0.0001, -0.0002]})
    q = np.array([13 * 3_600_000, 8 * 3_600_000, 33 * 3_600_000 + 16 * 60_000 + 300_000 * 3])
    f = deriv_features(q, {"oi": [oi], "fund": fund})
    # at 13:00 the newest usable snapshot is 12:55 (row 155); one hour earlier is row 143
    assert np.isclose(f["f_oi_chg_1h"][0], np.log(1155 / 1143))
    assert np.isclose(f["f_fund"][0], -2.0)
    assert np.isclose(f["f_fund"][1], 1.0)        # funding stamped exactly at the close is not used yet
    assert np.isnan(f["f_oi_chg_1h"][2])          # open interest too old -> missing, not stale
    assert deriv_features(q).isna().all().all()   # no futures data at all


def limit_engine():
    cfg = Config(order_type="limit", fee=0.001, maker_fee=0.0005, slippage=0.001, entry_wait=2,
                 min_train=10**9, gate_min_samples=0, gate_z=-1e9)
    eng = Engine(cfg)
    eng.model.predict = lambda x: 1.0
    eng.gate.extend([1.0])
    return eng


def test_limit_order_fills_only_when_traded_through():
    eng, x = limit_engine(), np.zeros(3, np.float32)
    out = eng.on_bar("AAA", 300_000, 101, 99, 100, 2.5, events=[(0, 1, x)])   # buy limit at 100
    order = out["opened"][0]
    assert not order["filled"] and eng.balance == 10_000 and eng.equity() == 10_000
    assert eng.on_bar("AAA", 600_000, 102, 100, 101, 2.5)["filled"] == []     # only touched
    assert len(eng.on_bar("AAA", 900_000, 101, 99.5, 100.5, 2.5)["filled"]) == 1
    qty = order["qty"]
    assert abs(eng.balance - (10_000 - qty * 100 * 0.0005)) < 1e-9            # maker fee, no slippage
    t = eng.on_bar("AAA", 1_200_000, 106, 100, 105, 2.5)["closed"][0]         # target = limit order
    assert t["reason"] == "target" and t["entry"] == 100 and abs(t["exit"] - 105) < 1e-9
    assert abs(t["pnl"] - (qty * 5 - qty * 100 * 0.0005 - qty * 105 * 0.0005)) < 1e-9
    assert abs(eng.r[0] - (0.05 - 0.001) / 0.05) < 1e-9
    # a stop is a market order: taker fee and slippage
    eng.on_bar("AAA", 1_500_000, 101, 99, 100, 2.5, events=[(0, 1, x)])
    t = eng.on_bar("AAA", 1_800_000, 100.5, 94, 95, 2.5)["closed"][0]         # filled and stopped
    assert t["reason"] == "stop" and abs(t["exit"] - 95 * 0.999) < 1e-9
    assert abs(eng.r[1] - (-0.05 - 0.0005 - 0.002) / 0.05) < 1e-9


def test_limit_order_expires_unfilled():
    eng, x = limit_engine(), np.zeros(3, np.float32)
    eng.on_bar("AAA", 300_000, 101, 99, 100, 2.5, events=[(0, -1, x)])        # sell limit at 100
    assert eng.on_bar("AAA", 600_000, 100, 98, 99, 2.5)["cancelled"] == []
    assert len(eng.on_bar("AAA", 900_000, 99, 97, 98, 2.5)["cancelled"]) == 1
    assert eng.open["AAA"] == [] and eng.r == [] and eng.trades == []
    assert eng.balance == 10_000 and eng.n_unfilled == 1
