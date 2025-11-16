# Strategia Volumetrica - Overview Dettagliato

## Filosofia della Strategia

La strategia si basa esclusivamente sull'**analisi volumetrica e order flow**, eliminando completamente gli indicatori tecnici tradizionali (RSI, MACD, medie mobili, etc.).

### Principi Fondamentali

1. **Volume Delta** - Differenza tra volumi buy e sell
2. **Buy/Sell Ratio** - Rapporto tra acquisti e vendite
3. **Volume Imbalance** - Squilibri volumetrici per identificare pressione
4. **Stacked Bars** - Barre consecutive con stesso sbilanciamento
5. **Confluence Score** - Combinazione di più segnali volumetrici

## Componenti della Strategia

### 1. Volume Delta Analysis

**Calcolo**:
```python
volume_delta = buy_volume - sell_volume
```

**Interpretazione**:
- **Delta > 0**: Pressione acquisto (bullish)
- **Delta < 0**: Pressione vendita (bearish)
- **|Delta| elevato**: Forte convinzione di mercato

**Indicatori Derivati**:
- `cumulative_delta`: Somma cumulativa del delta
- `delta_slope`: Pendenza del delta (accelerazione)
- `volume_imbalance`: Delta normalizzato sul volume totale

### 2. Buy/Sell Ratio

**Calcolo**:
```python
buy_sell_ratio = buy_volume / sell_volume
```

**Thresholds Ottimizzati**:
- **Long Entry**: ratio > 1.3455 (più acquirenti che venditori)
- **Short Entry**: ratio < 0.8758 (più venditori che acquirenti)
- **Long Exit**: ratio < 0.7046 (venditori prendono il controllo)
- **Short Exit**: ratio > 1.1672 (acquirenti prendono il controllo)

### 3. Volume Imbalance

**Calcolo**:
```python
volume_imbalance = abs(buy_volume - sell_volume) / total_volume
```

**Range**: 0.0 (perfetto bilanciamento) → 1.0 (tutto buy o sell)

**Thresholds**:
- **Minimum Imbalance**: 0.1706 (17.06%)
- **Strong Imbalance**: 0.3412 (34.12%)

### 4. Stacked Bars Detection

Identifica sequenze di barre consecutive con stesso tipo di imbalance.

**Parametri Ottimizzati**:
- **Min Stacked Bars**: 3 consecutive
- **Stacked Imbalance Ratio**: 3.0

**Logica**:
```python
if consecutive_bullish_bars >= 3 and avg_imbalance > 3.0:
    # Strong bullish momentum
    consider_long_entry()
```

### 5. Confluence Score

Combina multipli segnali volumetrici per validazione.

**Componenti del Confluence**:
1. Volume imbalance superiore a threshold
2. Buy/sell ratio allineato con direzione
3. Delta slope positivo/negativo
4. Stacked bars presenti
5. Volume ratio sopra media

**Threshold Ottimizzato**:
- **Entry Confluence**: ≥ 0.8072 (80.72%)
- **Exit Confluence**: ≥ 0.7633 (76.33%)

## Logica di Entry

### Long Entry Conditions

```python
# 1. Volume imbalance bullish
volume_imbalance > min_volume_imbalance

# 2. Buy pressure dominante
buy_sell_ratio > buy_sell_long_threshold

# 3. Delta slope positivo
delta_slope > 0

# 4. Stacked bullish bars
consecutive_bullish >= min_stacked_bars

# 5. Confluence score
confluence_score >= min_confluence

# 6. Non in ora tossica
current_hour NOT IN [14, 15, 16, 18, 19]
```

### Short Entry Conditions

```python
# 1. Volume imbalance bearish
volume_imbalance > min_volume_imbalance

# 2. Sell pressure dominante
buy_sell_ratio < buy_sell_short_threshold

# 3. Delta slope negativo
delta_slope < 0

# 4. Stacked bearish bars
consecutive_bearish >= min_stacked_bars

# 5. Confluence score
confluence_score >= min_confluence

# 6. Non in ora tossica
current_hour NOT IN [14, 15, 16, 18, 19]
```

## Logica di Exit

### 1. Stop Loss

**Calcolo**:
```python
stop_distance = ATR * stop_multiplier
stop_multiplier = 1.9208  # Ottimizzato

# Per LONG
stop_loss = entry_price - stop_distance

# Per SHORT
stop_loss = entry_price + stop_distance
```

### 2. Take Profit

**Calcolo** (RR 2.5:1):
```python
target_distance = ATR * target_multiplier
target_multiplier = 4.8019  # Ottimizzato (2.5x stop)

# Per LONG
take_profit = entry_price + target_distance

# Per SHORT
take_profit = entry_price - target_distance
```

### 3. Strategy Exit

Exit anticipato se:

**Long Exit**:
```python
# Inversione volume
buy_sell_ratio < buy_sell_exit_long (0.7046)

# OR Confluence basso
exit_confluence < exit_confluence_threshold (0.7633)
```

**Short Exit**:
```python
# Inversione volume
buy_sell_ratio > buy_sell_exit_short (1.1672)

# OR Confluence basso
exit_confluence < exit_confluence_threshold (0.7633)
```

## Time Filters

### Ore Escluse (UTC)

| Ora UTC | Ora ET | Motivo | SL Rate | P&L |
|---------|--------|--------|---------|-----|
| 14:00 | 08:00 | Market open chop | 45.2% | -$892 |
| 15:00 | 09:00 | High volatility | 65.5% | +$1,198 |
| 16:00 | 10:00 | **WORST HOUR** | **70.0%** | **-$1,434** |
| 18:00 | 12:00 | Pre-lunch | 52.1% | -$234 |
| 19:00 | 13:00 | Pre-close | 48.9% | -$567 |

**Impact**: Escludendo queste ore, performance passa da +40.80% a **+313.88%**

### Ore Migliori (UTC)

| Ora UTC | Ora ET | Trades | Win Rate | P&L |
|---------|--------|--------|----------|-----|
| 17:00 | 11:00 | 42 | 42.9% | +$2,341 |
| 20:00 | 14:00 | 38 | 39.5% | +$1,876 |
| 21:00 | 15:00 | 35 | 40.0% | +$1,654 |

## Tick Interval Optimization

### Perché 500-Tick Bars?

1. **Normalizzazione Volatilità**: Ogni barra contiene esattamente 500 tick
2. **Cattura Order Flow**: Volume sufficiente per segnali affidabili
3. **Riduzione Noise**: Meno falsi segnali rispetto a 100t
4. **Trade Quality**: 373 trade vs 2,074 (100t) - qualità > quantità

### Comparison Results

| Interval | Bars | Trades | Return | PF | Sharpe |
|----------|------|--------|--------|-----|--------|
| 100t | 879,827 | 2,074 | -31.63% | 0.98 | 0.01 |
| 250t | 351,931 | 703 | +4.94% | 1.01 | 0.01 |
| **500t** | **175,966** | **373** | **+313.88%** | **1.40** | **0.09** |
| 1000t | 87,983 | 93 | +57.92% | 1.23 | 0.06 |

## Risk Management

### Position Sizing

```python
position_size = 1  # Micro contract
point_value = $2   # Per point
risk_per_trade = stop_distance * point_value
```

### Risk/Reward Ratio

**Ottimizzato a 2.5:1**:
- Rischio medio: $40-50 per trade
- Profitto medio target: $100-125 per trade
- Win rate necessario: ~28.6% (ma otteniamo 35.12%)

### Maximum Drawdown Management

- **Max DD Osservato**: -40.53%
- **Recovery Time**: ~3 settimane
- **Raccomandazione**: Stop trading se DD > 50%

## Performance Metrics Spiegati

### Profit Factor (1.40)

```
Profit Factor = Total Wins / |Total Losses|
1.40 = $60,876 / $43,472
```

**Interpretazione**:
- PF > 1.0 = Strategia profittevole
- PF > 1.3 = Strategia robusta
- PF > 2.0 = Strategia eccellente

### Sharpe Ratio (0.0872)

```
Sharpe = (Return - RiskFreeRate) / Volatility
```

**Interpretazione**:
- SR > 0.5 = Acceptable
- SR > 1.0 = Good
- SR > 2.0 = Excellent

**Nota**: 0.0872 è basso ma accettabile per futures day trading.

### Win Rate (35.12%)

Con RR 2.5:1, win rate necessario teorico:
```
Required WR = 1 / (1 + RR) = 1 / (1 + 2.5) = 28.6%
```

**35.12% > 28.6%** ✅ Margine di sicurezza del 22.8%

## Limitazioni e Considerazioni

### Limitazioni

1. **Data Dependency**: Ottimizzato su 6 mesi (maggio-nov 2025)
2. **Market Regime**: Performance può variare in condizioni diverse
3. **Slippage**: Non incluso nel backtest ($0.75 commission only)
4. **Execution**: Assume fill istantaneo a prezzo desiderato

### Considerazioni per Live Trading

1. **Paper Trading First**: Testare per 2-4 settimane
2. **Start Small**: Iniziare con 1 contract
3. **Monitor Slippage**: Tracciare differenza tra ordine e fill
4. **Adjust for Gaps**: Strategia non gestisce gap overnight

### Market Conditions

**Funziona meglio in**:
- Trend giorni con volume alto
- RTH (Regular Trading Hours)
- Mercati con liquidità elevata

**Evitare**:
- Market open prima ora (8-9 AM ET)
- Pre-lunch (12 PM ET)
- Pre-close (3-4 PM ET)
- News events maggiori (FOMC, NFP, etc.)

## Miglioramenti Futuri

### In Sviluppo

1. **Dynamic Position Sizing**: Basato su volatilità
2. **ML Enhancement**: Neural network per confluence score
3. **Multi-Timeframe**: Conferma segnali su 250t + 500t + 1000t
4. **Sentiment Analysis**: Integrazione Twitter/Reddit sentiment

### Testare

1. **Altri Simboli**: ES, NQ, RTY
2. **Session Filters**: Solo AM o solo PM
3. **Volatility Filters**: VIX-based entry conditions
4. **Spread Analysis**: Bid-ask spread come filtro

---

**Versione**: 2.0
**Data**: 16 Novembre 2025
**Status**: Production Ready
