from dataclasses import dataclass, field


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    timeframes: tuple = ("5m", "15m", "30m", "1h", "4h")
    base_tf: str = "5m"          # the bot decides on every closed candle of this timeframe
    atr_tf: str = "1h"           # ATR of this timeframe sizes the profit/stop barriers

    # Trade / label definition (triple barrier): TP and SL both sit at
    # barrier_atr * ATR from entry; after `horizon` base candles the trade times out.
    barrier_atr: float = 2.0
    horizon: int = 96            # 96 x 5m = 8 hours

    # Costs (per side, as a fraction of notional)
    fee: float = 0.0004
    slippage: float = 0.0001

    # Paper account
    start_balance: float = 10_000.0
    risk_per_trade: float = 0.01  # fraction of equity lost when the stop is hit
    max_leverage: float = 3.0
    allow_short: bool = True

    # Learning
    lr: float = 0.0003
    l2: float = 1e-2
    min_train: int = 500          # labelled samples required before signals are produced
    min_edge: float = 0.02        # required probability margin above break-even

    # Self-check gate: only trade while the bot's own recent signals were profitable
    gate_window: int = 200
    gate_min_samples: int = 50
    gate_z: float = 2.0           # recent signal result must be this many std-errors above zero

    state_dir: str = "state"

    @property
    def round_trip_cost(self) -> float:
        return 2 * (self.fee + self.slippage)
