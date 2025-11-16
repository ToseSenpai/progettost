# Parametri Ottimali - Configurazione Strategia

## Parametri Baseline (Ottimizzati)

File di riferimento: `optimization_results/fixed_optimizer_500t_best_params_20251116_173532.json`

### Volume Parameters

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `min_volume_imbalance` | **0.1706** | Minimo squilibrio buy/sell (17.06%) |
| `strong_imbalance` | **0.3412** | Soglia per strong imbalance (34.12%) |
| `volume_ratio_threshold` | **2.0** | Rapporto volume rispetto a media |

**Spiegazione**:
- `min_volume_imbalance`: Se buy_volume - sell_volume < 17.06% del volume totale, il segnale non è abbastanza forte
- `strong_imbalance`: Sbilanciamento > 34.12% indica pressione molto forte
- `volume_ratio_threshold`: Il volume della barra deve essere almeno 2x la media

### Buy/Sell Ratio Parameters

| Parametro | Valore | Direzione | Descrizione |
|-----------|--------|-----------|-------------|
| `buy_sell_long_threshold` | **1.3455** | Entry LONG | Buy volume > 1.34x sell volume |
| `buy_sell_short_threshold` | **0.8758** | Entry SHORT | Sell volume > 1.14x buy volume |
| `buy_sell_exit_long` | **0.7046** | Exit LONG | Venditori prendono controllo |
| `buy_sell_exit_short` | **1.1672** | Exit SHORT | Acquirenti prendono controllo |

**Logica Entry**:
```python
# LONG: Quando acquirenti dominano
if buy_volume / sell_volume > 1.3455:
    consider_long()

# SHORT: Quando venditori dominano
if buy_volume / sell_volume < 0.8758:
    consider_short()
```

**Logica Exit**:
```python
# Exit LONG: Quando venditori iniziano a dominare
if position == LONG and buy_volume / sell_volume < 0.7046:
    exit_long()

# Exit SHORT: Quando acquirenti iniziano a dominare
if position == SHORT and buy_volume / sell_volume > 1.1672:
    exit_short()
```

### Delta Analysis Parameters

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `delta_slope_period` | **157** | Periodo per calcolo pendenza delta |

**Calcolo Delta Slope**:
```python
# Calcola pendenza del cumulative delta
delta_slope = (cumulative_delta[-1] - cumulative_delta[-delta_slope_period]) / delta_slope_period

# Segnale
if delta_slope > 0:  # Delta crescente
    bullish_signal()
else:  # Delta decrescente
    bearish_signal()
```

### Stacked Bars Parameters

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `min_stacked_bars` | **3** | Minimo barre consecutive stesso tipo |
| `stacked_imbalance_ratio` | **3.0** | Rapporto imbalance per stacking |

**Logica Stacked**:
```python
# Conta barre consecutive bullish
if volume_delta > 0:
    consecutive_bullish += 1
    consecutive_bearish = 0
else:
    consecutive_bearish += 1
    consecutive_bullish = 0

# Segnale forte se >= 3 barre consecutive
if consecutive_bullish >= 3:
    strong_bullish_momentum()
```

### Confluence Parameters

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `min_confluence` | **0.8072** | Minimo confluence score per entry (80.72%) |
| `exit_confluence_threshold` | **0.7633** | Soglia confluence per exit (76.33%) |

**Calcolo Confluence Score**:
```python
score = 0

# 1. Volume imbalance check (20%)
if volume_imbalance > min_volume_imbalance:
    score += 0.20

# 2. Buy/sell ratio aligned (25%)
if (is_long and buy_sell_ratio > buy_sell_long_threshold) or \
   (is_short and buy_sell_ratio < buy_sell_short_threshold):
    score += 0.25

# 3. Delta slope aligned (20%)
if (is_long and delta_slope > 0) or \
   (is_short and delta_slope < 0):
    score += 0.20

# 4. Stacked bars present (20%)
if consecutive_bars >= min_stacked_bars:
    score += 0.20

# 5. Volume above average (15%)
if current_volume > avg_volume * volume_ratio_threshold:
    score += 0.15

# Entry se score >= 0.8072
if score >= min_confluence:
    enter_trade()
```

### Risk/Reward Parameters

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `stop_multiplier` | **1.9208** | Moltiplicatore ATR per stop loss |
| `target_multiplier` | **4.8019** | Moltiplicatore ATR per take profit |
| `rr_ratio` | **2.5** | Risk/Reward ratio (target/stop) |

**Calcolo Stop e Target**:
```python
# ATR (Average True Range) come base
atr = calculate_atr(period=14)

# Stop Loss
stop_distance = atr * 1.9208
if position == LONG:
    stop_loss = entry_price - stop_distance
else:
    stop_loss = entry_price + stop_distance

# Take Profit (2.5x stop)
target_distance = atr * 4.8019  # = 1.9208 * 2.5
if position == LONG:
    take_profit = entry_price + target_distance
else:
    take_profit = entry_price - target_distance

# Verifica RR
rr_ratio = target_distance / stop_distance  # = 2.5
```

## Time Filters

### Ore Escluse (UTC)

```python
TOXIC_HOURS = [14, 15, 16, 18, 19]

def is_toxic_hour(timestamp):
    hour = timestamp.hour
    return hour in TOXIC_HOURS
```

| Ora UTC | Ora ET | Motivo | Action |
|---------|--------|--------|--------|
| 14 | 08:00 | Market open volatility | **SKIP** |
| 15 | 09:00 | Post-open adjustment | **SKIP** |
| 16 | 10:00 | **Worst hour** (70% SL) | **SKIP** |
| 18 | 12:00 | Pre-lunch thin volume | **SKIP** |
| 19 | 13:00 | Pre-close volatility | **SKIP** |

### Regular Trading Hours (RTH)

```python
def is_rth(timestamp):
    """Check if timestamp is during RTH."""
    hour = timestamp.hour

    # RTH: 13:30 - 20:00 UTC (8:30 AM - 3:00 PM ET)
    return 13 <= hour < 20
```

## Position Sizing

### Fixed Position Size

```python
POSITION_SIZE = 1  # Micro contract
POINT_VALUE = 2    # $ per point
```

### Risk per Trade

```python
risk_per_trade = stop_distance * POINT_VALUE

# Esempio:
# stop_distance = 20 points (ATR * 1.9208)
# risk = 20 * $2 = $40 per trade
```

### Capitale e Commissioni

```python
INITIAL_CAPITAL = 2000    # $2,000
COMMISSION = 0.75         # $0.75 per trade (round trip)
```

## File di Configurazione

### JSON Format

Salva questi parametri in `config/strategy_params.json`:

```json
{
  "volume": {
    "min_volume_imbalance": 0.17061609467522706,
    "strong_imbalance": 0.34123218935045413,
    "volume_ratio_threshold": 2.0
  },
  "buy_sell_ratio": {
    "long_threshold": 1.3454615678670163,
    "short_threshold": 0.8758497325476865,
    "exit_long": 0.7045741757098455,
    "exit_short": 1.1671734376015819
  },
  "delta": {
    "slope_period": 157
  },
  "stacked_bars": {
    "min_bars": 3,
    "imbalance_ratio": 3.0
  },
  "confluence": {
    "min_entry": 0.8071511679647423,
    "exit_threshold": 0.7633359044369274
  },
  "risk_reward": {
    "stop_multiplier": 1.9207625277570342,
    "target_multiplier": 4.8019063193925855,
    "rr_ratio": 2.5
  },
  "time_filters": {
    "toxic_hours_utc": [14, 15, 16, 18, 19],
    "rth_start_utc": 13,
    "rth_end_utc": 20
  },
  "trading": {
    "position_size": 1,
    "point_value": 2,
    "initial_capital": 2000,
    "commission": 0.75
  }
}
```

### Python Config Class

```python
from dataclasses import dataclass

@dataclass
class StrategyConfig:
    # Volume
    min_volume_imbalance: float = 0.1706
    strong_imbalance: float = 0.3412
    volume_ratio_threshold: float = 2.0

    # Buy/Sell Ratio
    buy_sell_long_threshold: float = 1.3455
    buy_sell_short_threshold: float = 0.8758
    buy_sell_exit_long: float = 0.7046
    buy_sell_exit_short: float = 1.1672

    # Delta
    delta_slope_period: int = 157

    # Stacked Bars
    min_stacked_bars: int = 3
    stacked_imbalance_ratio: float = 3.0

    # Confluence
    min_confluence: float = 0.8072
    exit_confluence_threshold: float = 0.7633

    # Risk/Reward
    stop_multiplier: float = 1.9208
    target_multiplier: float = 4.8019

    # Trading
    position_size: int = 1
    commission: float = 0.75
```

## Validazione Parametri

### Controlli Pre-Trade

```python
def validate_parameters(params):
    """Valida parametri prima di iniziare trading."""

    # 1. RR ratio check
    rr = params.target_multiplier / params.stop_multiplier
    assert 2.0 <= rr <= 3.0, f"RR ratio {rr:.2f} fuori range 2.0-3.0"

    # 2. Confluence threshold
    assert 0.7 <= params.min_confluence <= 0.9, \
        "Confluence troppo basso (<0.7) o alto (>0.9)"

    # 3. Volume imbalance
    assert 0.1 <= params.min_volume_imbalance <= 0.3, \
        "Volume imbalance fuori range 10-30%"

    # 4. Buy/sell ratio logic
    assert params.buy_sell_long_threshold > 1.0, \
        "Long threshold deve essere > 1.0"
    assert params.buy_sell_short_threshold < 1.0, \
        "Short threshold deve essere < 1.0"

    print("✅ Parametri validati correttamente")
```

## Sensitivity Analysis

### Parametri Più Sensibili

Dalla optimization history, i parametri con maggiore impact su performance:

1. **Time Filters** → Impact: **+773%** (da +40.80% a +313.88%)
2. **RR Ratio** → Impact: **+15%** (da 2.5:1 a 2.56:1)
3. **Min Confluence** → Impact: **+12%** (da 0.70 a 0.8072)
4. **Buy/Sell Thresholds** → Impact: **+8%**
5. **Volume Imbalance** → Impact: **+5%**

### Parametri Meno Sensibili

Questi parametri hanno minor impatto:

- `delta_slope_period` (150-165 simile performance)
- `stacked_imbalance_ratio` (2.5-3.5 simile)
- `volume_ratio_threshold` (1.5-2.5 simile)

## Raccomandazioni

### Per Beginners

Usare parametri conservativi:
```python
min_confluence = 0.85  # Più restrittivo (default 0.8072)
rr_ratio = 3.0         # Più conservativo (default 2.5)
```

### Per Advanced

Testare variazioni:
```python
# Test 1: Confluence più basso per più trade
min_confluence = 0.75

# Test 2: RR più alto per win rate più basso
rr_ratio = 3.5

# Test 3: Solo trade alta convinzione
min_volume_imbalance = 0.25
```

---

**Ultima Modifica**: 16 Novembre 2025
**Versione**: 2.0 (Ottimizzato + RR 2.5:1)
**File Origine**: `fixed_optimizer_500t_best_params_20251116_173532.json`
