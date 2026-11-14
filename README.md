# XBOT – zelflerende paper trading bot

Een bot die met **nepgeld** handelt en continu bijleert van candles op
**5m, 15m, 30m, 1H en 4H**. Geen API-key nodig: de candles komen van de publieke
Binance-marktdata-API. Er worden nooit echte orders geplaatst.

## Starten

```bash
pip install -r requirements.txt

python -m xbot backtest --days 60   # historie afspelen: bot leert en handelt onderweg
python -m xbot run                  # live paper trading (blijft draaien, elke 5 min)
python -m xbot status               # resultaten + laatste trades van de live bot
```

Opties (vóór het subcommando): `--symbol ETHUSDT`, `--balance 5000`, `--risk 0.005`,
`--no-short`, `--state-dir map`. Alle overige instellingen staan in `xbot/config.py`.

`run` leert bij de eerste start eerst 60 dagen historie (zonder te handelen) en gaat
daarna live. Model, positie en trades worden bewaard in `state/` (`state.json`,
`trades.csv`), dus stoppen en herstarten kan gewoon.

## Hoe het werkt

1. **Candles uitlezen** – per timeframe de gesloten candles (`xbot/data.py`).
2. **Analyseren** – per timeframe 13 kenmerken (`xbot/features.py`): RSI, ADX, +DI/−DI,
   volume t.o.v. gemiddelde, volumetrend, afstand tot EMA50, EMA20/50-kruising, MACD,
   Bollinger %B, ATR%, momentum (1 en 3 candles) en candle-body. Plus tijdstip van de dag:
   67 kenmerken in totaal. Hogere timeframes tellen pas mee als hun candle gesloten is.
3. **Voorspellen** – een online logistisch model (`xbot/model.py`) schat de kans dat de prijs
   eerst `2 × ATR(1H)` stijgt vóórdat hij evenveel daalt (binnen 8 uur).
4. **Handelen** – long/short alleen als die kans hoger is dan het break-even-punt inclusief
   fees en slippage, plus een marge. Stop-loss en take-profit liggen op dezelfde afstand;
   de positiegrootte is zo dat een stop 1% van de equity kost.
5. **Leren** – zodra de uitkomst van een eerdere candle bekend is, wordt het model daarmee
   bijgewerkt. Dat gebeurt bij elke candle, ook als er niet gehandeld wordt.
6. **Zelfcontrole** – de bot houdt bij wat zijn laatste 200 signalen ná kosten opgeleverd
   zouden hebben en opent alleen trades zolang dat aantoonbaar positief is ("gate").

## Eerlijke verwachting

Dit is een leer- en experimenteerplatform, geen geldmachine. In een backtest over
120 dagen BTCUSDT (tot 14 nov 2026) vond het model **geen voorspelkracht**: accuracy rond
48%, resultaat −5% bij 108 trades. Standaardindicatoren hadden in die periode geen
stabiel verband met de koers (verbanden keerden om tussen de eerste en tweede helft).
Kijk bij `status` vooral naar de log-loss: pas onder 0,693 doet het model het beter dan
een muntje opgooien. Gebruik dit niet met echt geld.

## Tests

```bash
pip install pytest && python -m pytest
```
