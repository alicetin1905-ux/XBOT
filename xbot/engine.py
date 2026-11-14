"""The learning + trading loop, shared by backtest and live paper trading.

For every closed base candle the engine:
  1. lets the broker check the open position (stop / target / timeout),
  2. resolves older samples whose outcome is now known and learns from them,
  3. predicts the current candle and, if the edge is big enough, opens a paper trade,
  4. stores the current candle as a pending sample to learn from later.
"""
import json
import os
from collections import deque

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .config import Config
from .features import feature_columns
from .model import OnlineLogit


class Engine:
    def __init__(self, cfg: Config, n_features: int):
        self.cfg = cfg
        self.model = OnlineLogit(n_features, cfg.lr, cfg.l2)
        self.broker = PaperBroker(cfg.start_balance, cfg.fee, cfg.slippage)
        self.pending = []                       # samples waiting for their outcome
        self.gate = deque(maxlen=cfg.gate_window)  # net result (in R) of recent signals
        self.scores = deque(maxlen=2000)        # (was prediction right?, log-loss) per sample
        self.last_ms = 0
        self.equity_curve = []

    # ---- learning -------------------------------------------------------
    def _resolve(self, high, low, close):
        cost = self.cfg.round_trip_cost
        keep = []
        for s in self.pending:
            up, dn = high >= s["up"], low <= s["dn"]
            s["bars_left"] -= 1
            if up and dn:        # both barriers inside one candle: outcome unknown
                y, w, move = None, 0.0, -s["b"] * s["signal"]  # count as a loss for a signal
            elif up:
                y, w, move = 1.0, 1.0, s["b"]
            elif dn:
                y, w, move = 0.0, 1.0, -s["b"]
            elif s["bars_left"] <= 0:
                move = close / s["entry"] - 1
                y, w = float(move > 0), min(1.0, abs(move) / s["b"])
            else:
                keep.append(s)
                continue
            if y is not None:
                p = min(max(s["p"], 1e-6), 1 - 1e-6)
                self.scores.append((float((p > 0.5) == (y > 0.5)),
                                    -(y * np.log(p) + (1 - y) * np.log(1 - p))))
                self.model.update(np.asarray(s["x"]), y, w)
            if s["signal"]:
                self.gate.append((s["signal"] * move - cost) / s["b"])
        self.pending = keep

    def gate_open(self):
        if len(self.gate) < self.cfg.gate_min_samples:
            return False
        g = np.asarray(self.gate)
        return float(g.mean() - self.cfg.gate_z * g.std() / np.sqrt(len(g))) > 0

    # ---- one candle -----------------------------------------------------
    def on_bar(self, close_ms, high, low, close, atr, x, trade=True):
        cfg = self.cfg
        closed = self.broker.on_bar(high, low, close, close_ms)
        self._resolve(high, low, close)
        self.model.observe(x)

        p = self.model.predict(x)
        b = cfg.barrier_atr * atr / close             # barrier distance as a fraction of price
        need = (b + cfg.round_trip_cost) / (2 * b) + cfg.min_edge  # break-even probability + margin
        signal = 0
        if self.model.updates >= cfg.min_train:
            if p > need:
                signal = 1
            elif 1 - p > need and cfg.allow_short:
                signal = -1

        opened = None
        if trade and signal and self.broker.position is None and self.gate_open():
            equity = self.broker.equity(close)
            qty = min(equity * cfg.risk_per_trade / (b * close), equity * cfg.max_leverage / close)
            self.broker.open(signal, qty, close, sl=close * (1 - signal * b),
                             tp=close * (1 + signal * b), bars=cfg.horizon,
                             time_ms=close_ms, prob=p)
            opened = self.broker.position

        self.pending.append({"x": [float(v) for v in x], "p": p, "entry": close, "b": b,
                             "up": close * (1 + b), "dn": close * (1 - b),
                             "bars_left": cfg.horizon, "signal": signal})
        self.last_ms = int(close_ms)
        self.equity_curve.append((int(close_ms), self.broker.equity(close)))
        return {"p": p, "need": need, "signal": signal, "opened": opened, "closed": closed}

    def run_frame(self, frame, trade=True, trade_only_last=False):
        """Feed every not-yet-seen row of a feature frame through the engine."""
        cols = feature_columns(frame)
        new = frame[frame["close_ms"] > self.last_ms]
        if self.last_ms and len(new) and new["close_ms"].iloc[0] - self.last_ms > 300_000 * 2:
            # we missed more candles than the fetched history covers: outcomes are unknowable
            if len(new) == len(frame):
                self.pending = []
        X = new[cols].to_numpy(float)
        meta = new[["close_ms", "high", "low", "close", "atr_ref"]].to_numpy(float)
        results = []
        for i in range(len(new)):
            t = trade and (not trade_only_last or i == len(new) - 1)
            results.append(self.on_bar(*meta[i], X[i], trade=t))
        return results

    # ---- reporting ------------------------------------------------------
    def stats(self):
        tr = self.broker.trades
        pnl = np.array([t["pnl"] for t in tr]) if tr else np.zeros(0)
        eq = np.array([e for _, e in self.equity_curve]) if self.equity_curve else np.array([self.cfg.start_balance])
        dd = float(((np.maximum.accumulate(eq) - eq) / np.maximum.accumulate(eq)).max())
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        sc = np.array(self.scores) if self.scores else np.zeros((0, 2))
        return {
            "equity": float(eq[-1]),
            "return_pct": float(eq[-1] / self.cfg.start_balance - 1) * 100,
            "trades": len(tr),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
            "profit_factor": float(wins / losses) if losses > 0 else None,
            "max_drawdown_pct": dd * 100,
            "samples_learned": self.model.updates,
            "accuracy_recent": float(sc[:, 0].mean()) if len(sc) else None,
            "logloss_recent": float(sc[:, 1].mean()) if len(sc) else None,  # 0.693 = coin flip
            "gate_signals": len(self.gate),
            "gate_avg_R": float(np.mean(self.gate)) if self.gate else None,
            "gate_open": self.gate_open(),
        }

    # ---- persistence ----------------------------------------------------
    def save(self):
        d = self.cfg.state_dir
        os.makedirs(d, exist_ok=True)
        state = {"model": self.model.to_dict(), "broker": self.broker.to_dict(),
                 "pending": self.pending, "gate": list(self.gate),
                 "scores": list(self.scores), "last_ms": self.last_ms,
                 "equity_curve": self.equity_curve[-5000:]}
        tmp = os.path.join(d, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(d, "state.json"))
        if self.broker.trades:
            t = pd.DataFrame(self.broker.trades)
            for c in ("entry_ms", "exit_ms"):
                t[c] = pd.to_datetime(t[c], unit="ms", utc=True)
            t.to_csv(os.path.join(d, "trades.csv"), index=False)

    @classmethod
    def load(cls, cfg: Config):
        path = os.path.join(cfg.state_dir, "state.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            s = json.load(f)
        e = cls(cfg, s["model"]["n"])
        e.model = OnlineLogit.from_dict(s["model"])
        e.broker = PaperBroker.from_dict(s["broker"])
        e.pending = s["pending"]
        e.gate.extend(s["gate"])
        e.scores.extend(tuple(x) for x in s["scores"])
        e.last_ms = s["last_ms"]
        e.equity_curve = [tuple(x) for x in s["equity_curve"]]
        return e
