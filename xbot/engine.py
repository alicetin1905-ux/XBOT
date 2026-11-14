"""The learning + paper-trading loop, shared by backtest and live trading.

Every setup the rules find becomes a *candidate*: a virtual trade with stop, target and
timeout. All candidates are followed to their outcome and become training data, whether
or not they were really traded. A candidate becomes a real paper trade when the model
expects enough profit and the self-check gate is open.
"""
import json
import os
import pickle
from collections import deque

import numpy as np
import pandas as pd

from .config import Config
from .model import SetupModel
from .setups import SETUPS

HOUR = 3_600_000


class Engine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = SetupModel()
        self.balance = cfg.start_balance
        self.open = {}            # symbol -> candidates still running (real trades have qty > 0)
        self.X, self.r, self.t = [], [], []   # resolved candidates = training data
        self.gate = deque(maxlen=cfg.gate_window)  # net R of the model's recent signals
        self.log = []             # resolved candidates that had a prediction (for reporting)
        self.trades = []
        self.last_ms = {}
        self.last_price = {}
        self.trend = {}           # symbol -> +1 uptrend, -1 downtrend, 0 sideways (30m/1h)
        self.equity_curve = []
        self.next_train_ms = 0
        self.trade_from_ms = 0    # no real trades before this time (learning only)
        self.n_filled = self.n_unfilled = 0   # candidates whose entry order did / did not fill
        self._dirty = True

    # ---- account --------------------------------------------------------
    def positions(self):
        return [(s, e) for s, lst in self.open.items() for e in lst if e["qty"] > 0]

    def equity(self):
        return self.balance + sum(e["side"] * e["qty"] * (self.last_price[s] - e["entry_px"])
                                  for s, e in self.positions() if e["filled"])

    def _costs(self, reason=None):
        """(fee rate, slippage) of the entry order, or of the exit order for `reason`."""
        cfg = self.cfg
        if cfg.order_type == "limit" and reason in (None, "target"):
            return cfg.maker_fee, 0.0
        return cfg.fee, cfg.slippage

    def gate_open(self):
        if len(self.gate) < self.cfg.gate_min_samples:
            return False
        g = np.asarray(self.gate)
        return float(g.mean() - self.cfg.gate_z * g.std() / np.sqrt(len(g))) > 0

    # ---- learning -------------------------------------------------------
    def retrain(self, now_ms):
        if len(self.r) < self.cfg.min_train:
            return False
        start = now_ms - self.cfg.train_window_days * 24 * HOUR
        i = int(np.searchsorted(np.asarray(self.t), start))
        self.X, self.r, self.t = self.X[i:], self.r[i:], self.t[i:]
        self.model.fit(np.asarray(self.X, np.float32), np.asarray(self.r))
        self.next_train_ms = now_ms + int(self.cfg.retrain_hours * HOUR)
        self._dirty = True
        return True

    def _finish(self, sym, e, exit_price, reason, now_ms, out):
        (fee_in, slip_in), (fee_out, slip_out) = self._costs(), self._costs(reason)
        r = (e["side"] * (exit_price / e["entry"] - 1)
             - (fee_in + slip_in + fee_out + slip_out)) / e["b"]
        self.X.append(e["x"]); self.r.append(r); self.t.append(e["t"])
        self._dirty = True
        if e["p"] is not None:
            self.log.append((e["t"], sym, e["setup"], e["side"], e["p"], e["signal"], r))
        if e["signal"]:
            self.gate.append(r)
        if e["qty"] > 0:
            exit_px = exit_price * (1 - e["side"] * slip_out)
            exit_fee = e["qty"] * exit_px * fee_out
            gross = e["side"] * e["qty"] * (exit_px - e["entry_px"])
            self.balance += gross - exit_fee
            trade = {"symbol": sym, "setup": SETUPS[e["setup"]],
                     "side": "long" if e["side"] > 0 else "short",
                     "entry_ms": e["t"], "exit_ms": int(now_ms), "qty": e["qty"],
                     "entry": e["entry_px"], "exit": exit_px, "reason": reason,
                     "expected_r": e["p"],
                     "pnl": gross - exit_fee - e["qty"] * e["entry_px"] * fee_in,
                     "balance": self.balance}
            self.trades.append(trade)
            out["closed"].append(trade)

    # ---- one candle of one symbol ---------------------------------------
    def on_bar(self, sym, close_ms, high, low, close, atr, events=(), trade=True):
        cfg = self.cfg
        out = {"closed": [], "opened": [], "filled": [], "cancelled": [], "candidates": []}
        close_ms = int(close_ms)
        self.last_price[sym] = close
        self.last_ms[sym] = close_ms

        running = self.open.get(sym)
        if running:
            keep = []
            for e in running:
                s = e["side"]
                hit_sl = low <= e["sl"] if s > 0 else high >= e["sl"]
                if not e["filled"]:   # limit entry: price has to trade through the order
                    if low < e["entry"] if s > 0 else high > e["entry"]:
                        e["filled"] = True
                        self.n_filled += 1
                        if e["qty"] > 0:
                            self.balance -= e["qty"] * e["entry_px"] * self._costs()[0]
                            out["filled"].append(e)
                        if hit_sl:    # filled and stopped within the same candle
                            self._finish(sym, e, e["sl"], "stop", close_ms, out)
                        else:
                            keep.append(e)
                    else:
                        e["wait"] -= 1
                        if e["wait"] > 0:
                            keep.append(e)
                        else:         # not filled in time: order cancelled, nothing to learn
                            self.n_unfilled += 1
                            if e["qty"] > 0:
                                out["cancelled"].append(e)
                    continue
                hit_tp = high >= e["tp"] if s > 0 else low <= e["tp"]
                e["bars_left"] -= 1
                if hit_sl:    # stop and target in the same candle: assume the stop came first
                    self._finish(sym, e, e["sl"], "stop", close_ms, out)
                elif hit_tp:
                    self._finish(sym, e, e["tp"], "target", close_ms, out)
                elif e["bars_left"] <= 0:
                    self._finish(sym, e, close, "timeout", close_ms, out)
                else:
                    keep.append(e)
            self.open[sym] = keep

        if close_ms >= self.next_train_ms:
            self.retrain(close_ms)

        for setup, side, x in events:
            p = self.model.predict(x)
            b = cfg.barrier_atr * atr / close       # barrier distance as a fraction of price
            signal = bool(p is not None and p > cfg.min_edge_r
                          and (side > 0 or cfg.allow_short))
            limit = cfg.order_type == "limit"
            fee_in, slip_in = self._costs()
            qty, entry_px = 0.0, close * (1 + side * slip_in)
            real = self.positions()
            if (trade and signal and close_ms >= self.trade_from_ms and self.gate_open()
                    and len(real) < cfg.max_positions and all(s != sym for s, _ in real)):
                eq = self.equity()
                qty = min(eq * cfg.risk_per_trade / (b * close), eq * cfg.max_leverage / close)
                if not limit:
                    self.balance -= qty * entry_px * fee_in
            self.n_filled += not limit
            e = {"x": [float(v) for v in x], "setup": int(setup), "side": int(side),
                 "entry": close, "b": b, "sl": close * (1 - side * b),
                 "tp": close * (1 + side * b), "bars_left": cfg.horizon, "p": p,
                 "signal": signal, "t": close_ms, "qty": qty, "entry_px": entry_px,
                 "filled": not limit, "wait": cfg.entry_wait if limit else 0}
            self.open.setdefault(sym, []).append(e)
            out["candidates"].append(e)
            if qty > 0:
                out["opened"].append(e)

        if close_ms % HOUR == 0 and close_ms >= self.trade_from_ms and (
                not self.equity_curve or self.equity_curve[-1][0] != close_ms):
            self.equity_curve.append((close_ms, self.equity()))
        return out

    def run(self, prepared, trade=True, trade_only_last=False, until_ms=None):
        """Feed all unseen candles of all symbols through the engine in time order.

        prepared: {symbol: (bars, events)} from setups.prepare_symbol.
        """
        syms = list(prepared)
        parts = []
        for k, s in enumerate(syms):
            ms = prepared[s][0][:, 0]
            rows = np.nonzero((ms > self.last_ms.get(s, 0)) & (ms < (until_ms or np.inf)))[0]
            parts.append(np.column_stack([ms[rows], np.full(len(rows), k), rows]))
        allrows = np.concatenate(parts) if parts else np.zeros((0, 3))
        if not len(allrows):
            return []
        allrows = allrows[np.lexsort((allrows[:, 1], allrows[:, 0]))]
        latest = allrows[-1, 0]
        results = []
        for ms, k, row in allrows:
            s = syms[int(k)]
            bars, events = prepared[s]
            ev = events.get(int(row), ())
            self.trend[s] = int(bars[int(row), 5])
            if not ev and not self.open.get(s) and ms % HOUR:
                self.last_ms[s] = int(ms)
                self.last_price[s] = bars[int(row), 3]
                continue
            t = trade and (not trade_only_last or ms == latest)
            res = self.on_bar(s, *bars[int(row), :5], events=ev, trade=t)
            if any(res.values()):
                results.append((s, res))
        return results

    # ---- reporting ------------------------------------------------------
    def stats(self):
        pnl = np.array([t["pnl"] for t in self.trades])
        eq = np.array([e for _, e in self.equity_curve] or [self.cfg.start_balance])
        peak = np.maximum.accumulate(eq)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        equity = self.equity()
        return {
            "equity": equity,
            "return_pct": (equity / self.cfg.start_balance - 1) * 100,
            "trades": len(pnl),
            "win_rate": float((pnl > 0).mean()) if len(pnl) else None,
            "profit_factor": float(wins / losses) if losses > 0 else None,
            "max_drawdown_pct": float(((peak - eq) / peak).max()) * 100,
            "fill_rate": self.n_filled / max(1, self.n_filled + self.n_unfilled),
            "samples": len(self.r),
            "model_trained_on": self.model.n_train,
            "gate_signals": len(self.gate),
            "gate_avg_r": float(np.mean(self.gate)) if self.gate else None,
            "gate_open": self.gate_open(),
        }

    def setup_table(self, from_ms=0):
        """Per setup: all predicted candidates vs. the ones the model selected."""
        rows = []
        log = [l for l in self.log if l[0] >= from_ms]
        for k, name in list(enumerate(SETUPS)) + [(None, "totaal")]:
            sel = [l for l in log if k is None or l[2] == k]
            r_all = np.array([l[6] for l in sel])
            r_sig = np.array([l[6] for l in sel if l[5]])
            rows.append({"setup": name, "n": len(r_all),
                         "avg_r": float(r_all.mean()) if len(r_all) else None,
                         "n_selected": len(r_sig),
                         "avg_r_selected": float(r_sig.mean()) if len(r_sig) else None})
        return rows

    def prediction_quality(self, from_ms=0):
        """Correlation between predicted and realised R, and realised R per prediction quintile."""
        log = [l for l in self.log if l[0] >= from_ms]
        if len(log) < 50:
            return None
        p, r = np.array([l[4] for l in log]), np.array([l[6] for l in log])
        edges = np.quantile(p, [0.2, 0.4, 0.6, 0.8])
        q = np.searchsorted(edges, p)
        return {"corr": float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 else 0.0,
                "quintile_avg_r": [float(r[q == i].mean()) if (q == i).any() else None
                                   for i in range(5)]}

    # ---- persistence ----------------------------------------------------
    def save(self):
        d = self.cfg.state_dir
        os.makedirs(d, exist_ok=True)
        state = {"balance": self.balance, "open": self.open, "gate": list(self.gate),
                 "log": self.log[-5000:], "trades": self.trades, "last_ms": self.last_ms,
                 "last_price": self.last_price, "trend": self.trend, "equity_curve": self.equity_curve[-5000:],
                 "next_train_ms": self.next_train_ms, "trade_from_ms": self.trade_from_ms,
                 "n_filled": self.n_filled, "n_unfilled": self.n_unfilled,
                 "setups": list(SETUPS)}
        tmp = os.path.join(d, "state.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(d, "state.json"))
        if self._dirty:   # training data and model only change when a candidate resolves
            np.savez_compressed(os.path.join(d, "samples.npz"), X=np.asarray(self.X, np.float32),
                                r=np.asarray(self.r), t=np.asarray(self.t, np.int64))
            with open(os.path.join(d, "model.pkl"), "wb") as f:
                pickle.dump(self.model, f)
            self._dirty = False
        if self.trades:
            t = pd.DataFrame(self.trades)
            for c in ("entry_ms", "exit_ms"):
                t[c] = pd.to_datetime(t[c], unit="ms", utc=True)
            t.to_csv(os.path.join(d, "trades.csv"), index=False)

    @classmethod
    def load(cls, cfg: Config):
        d = cfg.state_dir
        path = os.path.join(d, "state.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            s = json.load(f)
        if s.get("setups") != list(SETUPS):
            print("Opgeslagen state hoort bij een oudere versie (andere setups): opnieuw leren.")
            return None
        e = cls(cfg)
        e.balance, e.open, e.trades = s["balance"], s["open"], s["trades"]
        e.gate.extend(s["gate"])
        e.log = [tuple(l) for l in s["log"]]
        e.last_ms, e.last_price = s["last_ms"], s["last_price"]
        for lst in e.open.values():      # state written before limit orders existed
            for c in lst:
                c.setdefault("filled", True)
                c.setdefault("wait", 0)
        e.trend = s.get("trend", {})
        e.n_filled, e.n_unfilled = s.get("n_filled", 0), s.get("n_unfilled", 0)
        e.equity_curve = [tuple(x) for x in s["equity_curve"]]
        e.next_train_ms, e.trade_from_ms = s["next_train_ms"], s["trade_from_ms"]
        z = np.load(os.path.join(d, "samples.npz"))
        e.X, e.r, e.t = z["X"].tolist(), z["r"].tolist(), z["t"].tolist()
        with open(os.path.join(d, "model.pkl"), "rb") as f:  # our own file, written by save()
            e.model = pickle.load(f)
        e._dirty = False
        return e
