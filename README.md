# XBOT – zelflerende paper trading bot

Een bot die met **nepgeld** handelt en blijft bijleren van candles op
**5m, 15m, 30m, 1H en 4H**, voor meerdere munten tegelijk. Geen API-key nodig: de candles
komen van de publieke Binance-marktdata-API. Er worden nooit echte orders geplaatst.

## Starten

```bash
pip install -r requirements.txt

python -m xbot backtest             # ontwikkel-backtest op 2 jaar historie (zonder holdout)
python -m xbot backtest --holdout   # eindtoets op de apart gehouden laatste 90 dagen
python -m xbot run                  # live paper trading (blijft draaien, elke 5 min)
python -m xbot status               # resultaten + laatste trades van de live bot
```

Opties (vóór het subcommando): `--symbols BTCUSDT,ETHUSDT`, `--balance 5000`,
`--risk 0.005`, `--no-short`, `--state-dir map`. Overige instellingen: `xbot/config.py`.

De eerste keer wordt de historie gedownload (±1 minuut per munt) en lokaal bewaard in
`data/`; daarna worden alleen nieuwe candles opgehaald. `run` leert eerst een jaar
historie zonder te handelen en gaat dan live. Model, posities en trades staan in `state/`,
dus stoppen en herstarten kan gewoon.

## Hoe het werkt

1. **Candles uitlezen** – per munt en timeframe de gesloten candles (`xbot/data.py`).
2. **Analyseren** – per timeframe 13 kenmerken (`xbot/features.py`): RSI, ADX, +DI/−DI,
   volume t.o.v. gemiddelde, volumetrend, afstand tot EMA50, EMA20/50-kruising, MACD,
   Bollinger %B, ATR%, momentum en candle-body. Hogere timeframes tellen pas mee als hun
   candle gesloten is.
3. **Setups zoeken** – vaste regels leveren kandidaat-trades (`xbot/setups.py`):
   - *breakout*: prijs breekt uit zijn 4-uurs kanaal, met een trendende 1H-chart mee;
   - *reversal*: 15m-RSI extreem terwijl de 1H-chart niet trendt;
   - *pullback*: dip tegen een 4H-trend in.
4. **Beoordelen** – een boommodel (gradient boosting, `xbot/model.py`) voorspelt per
   kandidaat het verwachte resultaat na kosten. Eén model voor alle munten.
5. **Handelen** – alleen kandidaten met genoeg verwachte winst worden een paper trade.
   Stop-loss en take-profit liggen op `2 × ATR(1H)`; een stop kost 1% van de equity;
   maximaal 3 posities tegelijk, één per munt.
6. **Leren** – élke kandidaat wordt gevolgd tot zijn uitkomst bekend is, ook als er niet
   op gehandeld is. Het model wordt periodiek opnieuw getraind op alle bekende uitkomsten
   van het afgelopen jaar.
7. **Zelfcontrole** – de bot handelt alleen zolang zijn eigen laatste 200 signalen na
   kosten aantoonbaar winstgevend waren ("gate").

## Eerlijk testen

De laatste 90 dagen zijn de **holdout**: die periode wordt bij het ontwikkelen en afstellen
nooit bekeken. `backtest` gebruikt alleen de periode daarvóór; `backtest --holdout` is de
eindtoets. Draai die zelden: elke keer dat je op basis van de holdout iets aanpast, is hij
minder waard. Binnen elke backtest voorspelt het model alleen met kennis van uitkomsten
die op dat moment al bekend waren.

De backtest toont per setup het gemiddelde resultaat van **alle** kandidaten naast dat van
de **door het model gekozen** kandidaten. Dat verschil is wat het model toevoegt.

## Resultaat tot nu toe (14 nov 2026)

Getest op 2 jaar historie van BTC, ETH, SOL, BNB en XRP (ruim 15.000 setups).

| | kandidaten | gem. R alle | door model gekozen | gem. R gekozen | trades | rendement |
|---|---|---|---|---|---|---|
| Ontwikkelperiode | 12.505 | −0,078 | 1.140 | −0,059 | 7 | −3,6% |
| Holdout (90 dagen) | 1.835 | −0,040 | 92 | −0,058 | 0 | 0,0% |

- De setups verliezen gemiddeld ongeveer de transactiekosten: vóór kosten is er dus vrijwel
  geen edge.
- Het model voegt geen betrouwbare selectie toe (correlatie voorspeld/werkelijk rond 0).
- De zelfcontrole doet wat hij moet doen: de bot bleef bijna volledig uit de markt.

Dit is dus een eerlijk testplatform, geen winstgevende strategie. Gebruik het niet met
echt geld.

## Tests

```bash
pip install pytest && python -m pytest
```
