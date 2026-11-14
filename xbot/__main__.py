"""Command line:  python -m xbot backtest | run | status"""
import argparse
import time
from datetime import datetime, timezone

from .config import Config
from .data import TF_MS, fetch_all
from .engine import Engine
from .features import build_features, feature_columns


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _print_stats(s):
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    print(f"  Equity            : {s['equity']:.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Trades            : {s['trades']}  | winrate {f(s['win_rate'], '.1%')}"
          f" | profit factor {f(s['profit_factor'], '.2f')}")
    print(f"  Max drawdown      : {s['max_drawdown_pct']:.2f}%")
    print(f"  Geleerde samples  : {s['samples_learned']}")
    print(f"  Voorspelkwaliteit : accuracy {f(s['accuracy_recent'], '.1%')}"
          f" | log-loss {f(s['logloss_recent'], '.4f')} (0.6931 = muntje opgooien)")
    print(f"  Zelfcontrole-gate : {'OPEN' if s['gate_open'] else 'DICHT'}"
          f" | {s['gate_signals']} signalen, gem. {f(s['gate_avg_R'], '+.3f')} R na kosten")


def _history_frame(cfg, days):
    print(f"Candles downloaden: {cfg.symbol} {', '.join(cfg.timeframes)} ({days} dagen)...")
    frame = build_features(fetch_all(cfg.symbol, cfg.timeframes, days=days), cfg.base_tf, cfg.atr_tf)
    cutoff = frame["close_ms"].iloc[-1] - int(days * 86_400_000)
    return frame[frame["close_ms"] > cutoff].reset_index(drop=True)


def cmd_backtest(cfg, args):
    frame = _history_frame(cfg, args.days)
    eng = Engine(cfg, len(feature_columns(frame)))
    eng.run_frame(frame, trade=True)
    first, last = frame["close"].iloc[0], frame["close"].iloc[-1]
    print(f"\nBacktest {_ts(frame['close_ms'].iloc[0])} -> {_ts(frame['close_ms'].iloc[-1])} UTC"
          f"  ({len(frame)} candles, bot leert onderweg)")
    _print_stats(eng.stats())
    print(f"  Buy & hold        : {(last / first - 1) * 100:+.2f}%")
    for t in eng.broker.trades[-args.show:]:
        print(f"    {_ts(t['entry_ms'])}  {t['side']:5s} {t['entry']:.2f} -> {t['exit']:.2f}"
              f"  {t['reason']:7s} pnl {t['pnl']:+.2f}")


def cmd_run(cfg, args):
    eng = Engine.load(cfg)
    if eng is None:
        frame = _history_frame(cfg, args.warmup_days)
        eng = Engine(cfg, len(feature_columns(frame)))
        eng.run_frame(frame, trade=False)  # learn from history, no trades
        eng.equity_curve = []
        eng.save()
        print(f"Voorgeleerd op {eng.model.updates} samples. Live paper trading start nu.")
    else:
        print(f"State geladen: {eng.model.updates} samples geleerd, balans {eng.broker.balance:.2f}")
    step = TF_MS[cfg.base_tf]
    while True:
        try:
            frame = build_features(fetch_all(cfg.symbol, cfg.timeframes, limit=300),
                                   cfg.base_tf, cfg.atr_tf)
            results = eng.run_frame(frame, trade=True, trade_only_last=True)
            if results:
                r, price = results[-1], frame["close"].iloc[-1]
                for res in results:
                    if res["closed"]:
                        t = res["closed"]
                        print(f"  SLUIT {t['side']} @ {t['exit']:.2f} ({t['reason']}) pnl {t['pnl']:+.2f}")
                if r["opened"]:
                    o = r["opened"]
                    print(f"  OPEN {'long' if o['side'] > 0 else 'short'} @ {o['entry']:.2f}"
                          f"  sl {o['sl']:.2f}  tp {o['tp']:.2f}")
                pos = eng.broker.position
                print(f"{_ts(eng.last_ms)}  prijs {price:.2f}  p(up) {r['p']:.3f}"
                      f" (nodig >{r['need']:.3f})  signaal {r['signal']:+d}"
                      f"  gate {'open' if eng.gate_open() else 'dicht'}"
                      f"  positie {('long' if pos['side'] > 0 else 'short') if pos else '-'}"
                      f"  equity {eng.broker.equity(price):.2f}", flush=True)
                eng.save()
        except Exception as exc:  # network hiccups must not kill a long-running bot
            print(f"Fout (probeer volgende candle opnieuw): {exc!r}", flush=True)
        if args.once:
            break
        time.sleep(step / 1000 - (time.time() % (step / 1000)) + 5)


def cmd_status(cfg, args):
    eng = Engine.load(cfg)
    if eng is None:
        print("Nog geen state. Start eerst: python -m xbot run")
        return
    print(f"{cfg.symbol} - laatste candle {_ts(eng.last_ms)} UTC")
    _print_stats(eng.stats())
    p = eng.broker.position
    if p:
        print(f"  Open positie      : {'long' if p['side'] > 0 else 'short'} {p['qty']:.5f}"
              f" @ {p['entry']:.2f}  sl {p['sl']:.2f}  tp {p['tp']:.2f}")
    for t in eng.broker.trades[-args.show:]:
        print(f"    {_ts(t['entry_ms'])}  {t['side']:5s} {t['entry']:.2f} -> {t['exit']:.2f}"
              f"  {t['reason']:7s} pnl {t['pnl']:+.2f}")


def main():
    ap = argparse.ArgumentParser(prog="xbot", description="Zelflerende paper trading bot")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--risk", type=float, default=0.01, help="risico per trade (fractie van equity)")
    ap.add_argument("--no-short", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="speel historie af; de bot leert en handelt onderweg")
    b.add_argument("--days", type=float, default=60)
    b.add_argument("--show", type=int, default=10, help="aantal laatste trades om te tonen")
    r = sub.add_parser("run", help="live paper trading (blijft draaien)")
    r.add_argument("--warmup-days", type=float, default=60)
    r.add_argument("--once", action="store_true", help="verwerk nieuwe candles en stop")
    s = sub.add_parser("status", help="toon resultaten van de live bot")
    s.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    cfg = Config(symbol=args.symbol, state_dir=args.state_dir, start_balance=args.balance,
                 risk_per_trade=args.risk, allow_short=not args.no_short)
    {"backtest": cmd_backtest, "run": cmd_run, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
