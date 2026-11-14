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
`--risk 0.005`, `--no-short`, `--market`, `--state-dir map`. Overige instellingen: `xbot/config.py`.

De eerste keer wordt de historie gedownload (±1 minuut per munt) en lokaal bewaard in
`data/`; daarna worden alleen nieuwe candles opgehaald. `run` leert eerst een jaar
historie zonder te handelen en gaat dan live. Model, posities en trades staan in `state/`,
dus stoppen en herstarten kan gewoon.

## Hoe het werkt

1. **Candles uitlezen** – per munt en timeframe de gesloten candles (`xbot/data.py`).
2. **Analyseren** – per timeframe 13 kenmerken (`xbot/features.py`): RSI, ADX, +DI/−DI,
   volume t.o.v. gemiddelde, volumetrend, afstand tot EMA50, EMA20/50-kruising, MACD,
   Bollinger %B, ATR%, momentum, candle-body en het aandeel agressieve kopers in het
   volume. Hogere timeframes tellen pas mee als hun candle gesloten is. Daarbij komen
   futures-kenmerken (`xbot/derivs.py`): verandering in open interest over 1, 4 en 24 uur
   en de funding rate. Historie komt uit de Binance-bulkbestanden, live uit de
   Binance-futures-API of, waar die niet bereikbaar is, OKX.
3. **Trend bepalen** – per munt op de 30m- en 1H-chart: *uptrend* als op beide EMA20 boven
   EMA50 ligt en +DI boven −DI, *downtrend* bij het omgekeerde, anders *zijwaarts*. De
   trend staat in de live-uitvoer en in `status`.
4. **Setups zoeken** – instappen komen van de 5m- en 15m-chart en gaan alleen met de trend
   mee (`xbot/setups.py`):
   - *breakout*: 5m-koers breekt uit zijn kanaal van 2 uur, in de richting van de trend;
   - *pullback*: 5m- of 15m-RSI zakt tegen de trend in;
   - *reversal*: 15m-RSI extreem, alleen als er geen trend is.
5. **Beoordelen** – een boommodel (gradient boosting, `xbot/model.py`) voorspelt per
   kandidaat het verwachte resultaat na kosten. Eén model voor alle munten.
6. **Handelen** – alleen kandidaten met genoeg verwachte winst worden een paper trade.
   De instap is een limit-order op de slotkoers van de signaalcandle; die telt alleen als
   gevuld wanneer de koers er binnen 3 candles doorheen handelt, anders vervalt hij. Ook
   het doel is een limit-order (maker-fee); stop en time-out zijn market-orders
   (taker-fee plus slippage). Met `--market` gaat alles via market-orders.
   Stop-loss en take-profit liggen op `2 × ATR(1H)`; een stop kost 1% van de equity;
   maximaal 3 posities tegelijk, één per munt.
7. **Leren** – élke kandidaat wordt gevolgd tot zijn uitkomst bekend is, ook als er niet
   op gehandeld is. Het model wordt periodiek opnieuw getraind op alle bekende uitkomsten
   van het afgelopen jaar.
8. **Zelfcontrole** – de bot handelt alleen zolang zijn eigen laatste 200 signalen na
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

Daarna zijn futures-kenmerken (open interest, funding) en de koop/verkoop-verhouding
toegevoegd. Op de ontwikkelperiode veranderde dat niets wezenlijks: gekozen kandidaten
−0,081 R tegenover −0,078 R voor alle kandidaten, 15 trades, −4,2%. De holdout is daarvoor
niet opnieuw gedraaid.

Vervolgens is de trend naar 30m/1H verplaatst en de instap naar 5m/15m, alleen met de
trend mee. Ontwikkelperiode: 15.076 kandidaten, gem. −0,060 R; door het model gekozen
−0,073 R; 44 trades, −8,2%. Ook dit gaf geen edge na kosten.

Met limit-orders dalen de kosten: alle kandidaten gemiddeld −0,040 R in plaats van −0,060 R,
97% van de instap-orders gevuld. Het blijft negatief; doordat de zelfcontrole vaker
openging werd er meer gehandeld: 226 trades, −11,5% (market-orders: 44 trades, −8,2%).

Dit is dus een eerlijk testplatform, geen winstgevende strategie. Gebruik het niet met
echt geld.

## Tests

```bash
pip install pytest && python -m pytest
```
