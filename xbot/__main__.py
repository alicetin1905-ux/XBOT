"""Command line:  python -m xbot backtest | run | status"""
import argparse
import time
from datetime import datetime, timezone

from .config import Config
from .data import TF_MS, fetch_recent, load_history
from .derivs import fetch_derivs_recent, load_derivs_history
from .engine import Engine
from .setups import prepare_symbol

DAY = 86_400_000


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _trends(eng):
    names = {1: "uptrend", -1: "downtrend", 0: "zijwaarts"}
    return ", ".join(f"{s.replace('USDT', '')} {names[t]}" for s, t in eng.trend.items())


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def _print_stats(eng, from_ms=0):
    s = eng.stats()
    print(f"  Equity            : {s['equity']:.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Trades            : {s['trades']}  | winrate {_f(s['win_rate'], '.1%')}"
          f" | profit factor {_f(s['profit_factor'], '.2f')}")
    print(f"  Max drawdown      : {s['max_drawdown_pct']:.2f}%")
    print(f"  Trainingsdata     : {s['samples']} afgeronde setups"
          f" (model getraind op {s['model_trained_on']})")
    print(f"  Zelfcontrole-gate : {'OPEN' if s['gate_open'] else 'DICHT'}"
          f" | {s['gate_signals']} signalen, gem. {_f(s['gate_avg_r'], '+.3f')} R na kosten")
    print("  Setups (R = resultaat na kosten, 1R = afstand tot de stop):")
    print("    setup       kandidaten  gem. R   door model gekozen  gem. R")
    for r in eng.setup_table(from_ms):
        print(f"    {r['setup']:10s} {r['n']:10d}  {_f(r['avg_r'], '+.3f'):>6s}"
              f"   {r['n_selected']:17d}  {_f(r['avg_r_selected'], '+.3f'):>6s}")
    q = eng.prediction_quality(from_ms)
    if q:
        print(f"  Voorspelkwaliteit : correlatie voorspeld/werkelijk {q['corr']:+.3f}")
        print("    werkelijk gem. R per voorspel-kwintiel (laag -> hoog): "
              + "  ".join(_f(v, "+.3f") for v in q["quintile_avg_r"]))


def _print_trades(trades, n):
    for t in trades[-n:]:
        print(f"    {_ts(t['entry_ms'])}  {t['symbol']:9s} {t['setup']:9s} {t['side']:5s}"
              f" {t['entry']:.4g} -> {t['exit']:.4g}  {t['reason']:7s} pnl {t['pnl']:+.2f}")


def _history(cfg, days):
    print(f"Candles laden: {', '.join(cfg.symbols)} | {', '.join(cfg.timeframes)} | {days:g} dagen")
    return {s: prepare_symbol(load_history(s, cfg.timeframes, days, cfg.cache_dir), cfg,
                              load_derivs_history(s, days, cfg.cache_dir))
            for s in cfg.symbols}


def _live_derivs(symbol):
    try:
        return fetch_derivs_recent(symbol)
    except Exception as exc:   # the bot keeps running on candle features alone
        print(f"  ({symbol}: futures-data niet beschikbaar: {exc!r})")
        return None


def cmd_backtest(cfg, args):
    prepared = _history(cfg, args.days)
    last = max(bars[-1, 0] for bars, _ in prepared.values())
    holdout_start = int(last - cfg.holdout_days * DAY)
    eng = Engine(cfg)
    if args.holdout:
        eng.trade_from_ms = report_from = holdout_start
        eng.run(prepared)
        print(f"\nHOLDOUT-test {_ts(holdout_start)} -> {_ts(last)} UTC"
              "  (nooit gebruikt bij het afstellen; draai dit zelden)")
    else:
        eng.run(prepared, until_ms=holdout_start)
        report_from = eng.log[0][0] if eng.log else holdout_start
        print(f"\nOntwikkel-backtest {_ts(report_from)} -> {_ts(holdout_start)} UTC"
              f"  (laatste {cfg.holdout_days:g} dagen blijven apart als holdout)")
    _print_stats(eng, report_from)
    bh = []
    for bars, _ in prepared.values():
        seg = bars[(bars[:, 0] >= report_from) & (bars[:, 0] <= (last if args.holdout else holdout_start))]
        if len(seg):
            bh.append(seg[-1, 3] / seg[0, 3] - 1)
    print(f"  Buy & hold (gem.) : {sum(bh) / len(bh) * 100:+.2f}%")
    _print_trades(eng.trades, args.show)


def cmd_run(cfg, args):
    eng = Engine.load(cfg)
    if eng is None:
        eng = Engine(cfg)
        eng.run(_history(cfg, args.warmup_days), trade=False)  # learn from history, no trades
        eng.trade_from_ms = max(eng.last_ms.values())
        eng.save()
        print(f"Voorgeleerd op {len(eng.r)} setups. Live paper trading start nu.")
    else:
        print(f"State geladen: {len(eng.r)} setups geleerd, balans {eng.balance:.2f}")
    step = TF_MS[cfg.base_tf] / 1000
    while True:
        try:
            prepared = {s: prepare_symbol(fetch_recent(s, cfg.timeframes), cfg, _live_derivs(s))
                        for s in cfg.symbols}
            n_cand = 0
            for sym, res in eng.run(prepared, trade=True, trade_only_last=True):
                n_cand += len(res["candidates"])
                for t in res["closed"]:
                    print(f"  SLUIT {sym} {t['side']} @ {t['exit']:.4g} ({t['reason']}) pnl {t['pnl']:+.2f}")
                for o in res["opened"]:
                    print(f"  OPEN  {sym} {'long' if o['side'] > 0 else 'short'} @ {o['entry_px']:.4g}"
                          f"  sl {o['sl']:.4g}  tp {o['tp']:.4g}  verwacht {o['p']:+.2f} R")
            pos = ", ".join(f"{s} {'long' if e['side'] > 0 else 'short'}" for s, e in eng.positions())
            print(f"{_ts(max(eng.last_ms.values()))}  trend [{_trends(eng)}]")
            print(f"                  nieuwe setups {n_cand}"
                  f"  gate {'open' if eng.gate_open() else 'dicht'}"
                  f"  posities [{pos or '-'}]  equity {eng.equity():.2f}", flush=True)
            eng.save()
        except Exception as exc:  # network hiccups must not kill a long-running bot
            print(f"Fout (probeer volgende candle opnieuw): {exc!r}", flush=True)
        if args.once:
            break
        time.sleep(step - (time.time() % step) + 5)


def cmd_status(cfg, args):
    eng = Engine.load(cfg)
    if eng is None:
        print("Nog geen state. Start eerst: python -m xbot run")
        return
    print(f"Laatste candle {_ts(max(eng.last_ms.values()))} UTC")
    print(f"  Trend (30m/1H)    : {_trends(eng)}")
    _print_stats(eng, eng.trade_from_ms)
    for s, e in eng.positions():
        print(f"  Open positie      : {s} {'long' if e['side'] > 0 else 'short'} {e['qty']:.5g}"
              f" @ {e['entry_px']:.4g}  sl {e['sl']:.4g}  tp {e['tp']:.4g}")
    _print_trades(eng.trades, args.show)


def main():
    d = Config()
    ap = argparse.ArgumentParser(prog="xbot", description="Zelflerende paper trading bot")
    ap.add_argument("--symbols", default=",".join(d.symbols), help="komma-gescheiden")
    ap.add_argument("--state-dir", default=d.state_dir)
    ap.add_argument("--balance", type=float, default=d.start_balance)
    ap.add_argument("--risk", type=float, default=d.risk_per_trade,
                    help="risico per trade (fractie van equity)")
    ap.add_argument("--no-short", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="speel historie af; de bot leert en handelt onderweg")
    b.add_argument("--days", type=float, default=730)
    b.add_argument("--holdout", action="store_true",
                   help="test op de apart gehouden laatste periode (eindtoets)")
    b.add_argument("--holdout-days", type=float, default=d.holdout_days)
    b.add_argument("--retrain-hours", type=float, default=168,
                   help="hertrain-interval in de backtest (live: elke 24 uur)")
    b.add_argument("--show", type=int, default=10, help="aantal laatste trades om te tonen")
    r = sub.add_parser("run", help="live paper trading (blijft draaien)")
    r.add_argument("--warmup-days", type=float, default=365)
    r.add_argument("--once", action="store_true", help="verwerk nieuwe candles en stop")
    s = sub.add_parser("status", help="toon resultaten van de live bot")
    s.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    cfg = Config(symbols=tuple(args.symbols.split(",")), state_dir=args.state_dir,
                 start_balance=args.balance, risk_per_trade=args.risk,
                 allow_short=not args.no_short)
    if args.cmd == "backtest":
        cfg.holdout_days, cfg.retrain_hours = args.holdout_days, args.retrain_hours
    {"backtest": cmd_backtest, "run": cmd_run, "status": cmd_status}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
