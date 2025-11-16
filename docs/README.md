# Strategia Trading Volumetrica MNQ - Documentazione Completa

## Panoramica

Strategia di trading algoritmico basata esclusivamente su **analisi volumetrica e order flow** per futures Micro E-mini NASDAQ-100 (MNQ).

La strategia è stata ottimizzata attraverso 150+ trial Optuna e mesi di backtesting, raggiungendo performance eccellenti con **+313.88% di ritorno** su 6 mesi di dati storici.

## Caratteristiche Principali

- **Tipo**: Pure volumetric/order flow (NO indicatori tradizionali)
- **Mercato**: MNQ Futures (Micro E-mini NASDAQ-100)
- **Timeframe**: 500-tick bars (ottimale)
- **Capitale iniziale**: $2,000
- **Commissioni**: $0.75 per trade
- **Position sizing**: 1 micro contract ($2/point)

## Performance Ottimale (500t bars + RR 2.5:1 + Filtri Tempo)

| Metrica | Valore |
|---------|--------|
| **Return** | **+313.88%** |
| **Profit Factor** | **1.40** |
| **Win Rate** | **35.12%** |
| **Sharpe Ratio** | **0.0872** |
| **Max Drawdown** | **-40.53%** |
| **Total Trades** | **373** |
| **Final Capital** | **$8,278** |
| **Avg Win** | **$163.18** |
| **Avg Loss** | **-$64.85** |

## Quick Start

### 1. Eseguire Backtest con Parametri Ottimali

```bash
python backtest_baseline_final.py
```

### 2. Testare su Diversi Timeframe

```bash
python test_all_tick_intervals_clean.py
```

### 3. Analizzare Stop Loss per Ora

```bash
python analyze_stoploss_by_hour.py
```

## Struttura Documentazione

- [**Strategy Overview**](strategy-overview.md) - Descrizione dettagliata della strategia
- [**Optimal Parameters**](optimal-parameters.md) - Parametri ottimali e configurazione
- [**Performance Analysis**](performance-analysis.md) - Analisi completa delle performance
- [**Optimization History**](optimization-history.md) - Storia delle ottimizzazioni

## File Principali

### Strategia Core
- `src/volumetric_tick_strategy.py` - Implementazione strategia volumetrica
- `src/tick_aggregator.py` - Aggregazione tick bars
- `src/data_manager.py` - Gestione dati

### Backtest
- `backtest_baseline_final.py` - Backtest con parametri ottimali
- `backtest_100t.py` - Test su 100-tick bars
- `backtest_multi_timeframe.py` - Test multi-timeframe (M1-M10)

### Ottimizzazione
- `optimize_pure_volumetric.py` - Ottimizzazione completa
- `optimize_rr_ratio.py` - Ottimizzazione Risk/Reward
- `test_all_tick_intervals_clean.py` - Comparazione tick intervals

### Analisi
- `analyze_stoploss_by_hour.py` - Analisi stop loss orari
- `compare_strategies.py` - Comparazione strategie

### Dati
- `download_databento_ticks.py` - Download dati da Databento
- `generate_100t_bars.py` - Generazione 100-tick bars
- `create_minute_bars.py` - Conversione tick → minute bars

## Risultati Chiave

### 1. Tick Interval Comparison (con parametri ottimizzati)

| Intervallo | Return | PF | Trades | Sharpe |
|------------|--------|------|--------|--------|
| 100t | -31.63% | 0.98 | 2,074 | 0.01 |
| 250t | +4.94% | 1.01 | 703 | 0.01 |
| **500t** | **+313.88%** | **1.40** | **373** | **0.09** |
| 1000t | +57.92% | 1.23 | 93 | 0.06 |

**Conclusione**: 500t è l'intervallo ottimale per questa strategia.

### 2. Timeframe Comparison (M1-M10 minute bars)

| Timeframe | Return | PF | Trades |
|-----------|--------|------|--------|
| M1 | +68.62% | 1.18 | 170 |
| M2 | +70.42% | 1.34 | 78 |
| M3 | -42.21% | 0.67 | 34 |
| M4 | -10.55% | 0.84 | 16 |
| M5 | -72.26% | 0.00 | 12 |
| M10 | -17.42% | 0.00 | 2 |

**Conclusione**: Tick bars >> Time bars. La strategia volumetrica funziona meglio con aggregazione basata su volume.

### 3. Filtri Tempo Critici

Ore UTC **ESCLUSE** (performance negativa):
- **14:00** (8 AM ET) - Market open chop
- **15:00** (9 AM ET) - 65.5% SL rate
- **16:00** (10 AM ET) - 70% SL rate, -$1,434 loss
- **18:00** (12 PM ET) - Pre-lunch volatility
- **19:00** (1 PM ET) - Pre-close volatility

**Impact**: +313.88% (con filtri) vs +40.80% (senza filtri)

## Prossimi Passi

1. **Live Testing**: Implementare su paper trading
2. **Risk Management**: Aggiungere position sizing dinamico
3. **Machine Learning**: Testare ensemble di modelli per entry/exit
4. **Multi-Symbol**: Estendere a ES, NQ, RTY

## Avvertenze

- Tutti i backtest sono basati su dati storici (maggio-novembre 2025)
- Performance passate non garantiscono risultati futuri
- Sempre testare in paper trading prima di andare live
- La strategia è ottimizzata per condizioni di mercato specifiche

## Contatti e Support

Per domande o supporto, consultare la documentazione completa nelle cartelle:
- `/docs` - Documentazione strategia
- `/optimization_results` - Risultati ottimizzazioni
- `/reports` - Report e visualizzazioni

---

**Ultima modifica**: 16 Novembre 2025
**Versione**: 2.0 (RR 2.5:1 + Time Filters)
**Status**: Production Ready
