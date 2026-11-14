"""Paper broker: one position at a time, with fees, slippage, stop, target and timeout."""


class PaperBroker:
    def __init__(self, balance=10_000.0, fee=0.0004, slippage=0.0001):
        self.balance = balance
        self.fee, self.slippage = fee, slippage
        self.position = None
        self.trades = []

    def equity(self, price):
        p = self.position
        return self.balance + (p["side"] * p["qty"] * (price - p["entry"]) if p else 0.0)

    def open(self, side, qty, price, sl, tp, bars, time_ms, prob=None):
        entry = price * (1 + side * self.slippage)
        self.balance -= qty * entry * self.fee
        self.position = {"side": side, "qty": qty, "entry": entry, "sl": sl, "tp": tp,
                         "bars_left": bars, "time_ms": time_ms, "prob": prob,
                         "entry_fee": qty * entry * self.fee}

    def on_bar(self, high, low, close, time_ms):
        """Check the open position against a new candle. Returns the closed trade or None."""
        p = self.position
        if not p:
            return None
        s = p["side"]
        hit_sl = low <= p["sl"] if s > 0 else high >= p["sl"]
        hit_tp = high >= p["tp"] if s > 0 else low <= p["tp"]
        p["bars_left"] -= 1
        if hit_sl:  # if stop and target fall in the same candle, assume the stop came first
            return self._close(p["sl"], "stop", time_ms)
        if hit_tp:
            return self._close(p["tp"], "target", time_ms)
        if p["bars_left"] <= 0:
            return self._close(close, "timeout", time_ms)
        return None

    def _close(self, price, reason, time_ms):
        p = self.position
        exit_price = price * (1 - p["side"] * self.slippage)
        exit_fee = p["qty"] * exit_price * self.fee
        gross = p["side"] * p["qty"] * (exit_price - p["entry"])
        self.balance += gross - exit_fee
        trade = {"entry_ms": p["time_ms"], "exit_ms": time_ms,
                 "side": "long" if p["side"] > 0 else "short", "qty": p["qty"],
                 "entry": p["entry"], "exit": exit_price, "reason": reason, "prob": p["prob"],
                 "pnl": gross - exit_fee - p["entry_fee"], "balance": self.balance}
        self.trades.append(trade)
        self.position = None
        return trade

    def to_dict(self):
        return {"balance": self.balance, "fee": self.fee, "slippage": self.slippage,
                "position": self.position, "trades": self.trades}

    @classmethod
    def from_dict(cls, d):
        b = cls(d["balance"], d["fee"], d["slippage"])
        b.position, b.trades = d["position"], d["trades"]
        return b
