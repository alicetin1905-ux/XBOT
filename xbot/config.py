from dataclasses import dataclass


@dataclass
class Config:
    symbols: tuple = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT")
    timeframes: tuple = ("5m", "15m", "30m", "1h", "4h")
    base_tf: str = "5m"          # setups are detected on every closed candle of this timeframe
    atr_tf: str = "1h"           # ATR of this timeframe sizes the profit/stop barriers

    # Trade definition: TP and SL both sit at barrier_atr * ATR from entry;
    # after `horizon` base candles the trade times out.
    barrier_atr: float = 2.0
    horizon: int = 96            # 96 x 5m = 8 hours

    # Setup detection
    breakout_lookback: int = 24  # base candles for the high/low channel (2 hours)
    fakeout_lookback: int = 24   # base candles (2 hours) of sideways movement before a fakeout
    fakeout_range_atr: float = 1.0  # max height of that sideways range, in 1h ATRs
    fakeout_depth_atr: float = 1.5  # max depth of that poke, in 5m ATRs
    setup_cooldown: int = 12     # a setup must have been absent this many candles to fire again

    # Orders. "limit": entry is a limit order at the signal candle's close that must be
    # traded through within `entry_wait` candles, otherwise it is cancelled; the target is
    # a limit order too. Stops and timeouts are always market orders.
    order_type: str = "limit"    # or "market"
    entry_wait: int = 3

    # Costs (per side, as a fraction of notional)
    fee: float = 0.0004          # market orders (taker)
    maker_fee: float = 0.0002    # limit orders (maker)
    slippage: float = 0.0001     # market orders only

    # Paper account
    start_balance: float = 10_000.0
    risk_per_trade: float = 0.01  # fraction of equity lost when the stop is hit
    max_leverage: float = 1.0     # max notional per position, as a multiple of equity
    max_positions: int = 3        # across all symbols (one per symbol)
    allow_short: bool = True

    # Learning
    retrain_hours: float = 24     # the model is refitted on all known outcomes this often
    train_window_days: float = 365
    min_train: int = 1000         # resolved setups required before the first model is fitted
    min_edge_r: float = 0.05      # predicted net result (in R) required to take a setup

    # Self-check gate: only trade while the bot's own recent signals were profitable
    gate_window: int = 200
    gate_min_samples: int = 50
    gate_z: float = 1.0           # recent signal result must be this many std-errors above zero

    # Evaluation: the most recent days are kept apart and never used while developing
    holdout_days: float = 90

    state_dir: str = "state"
    cache_dir: str = "data"
