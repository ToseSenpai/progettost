# ProjectX Trading Bot

Sistema di trading automatico per TopStepX che utilizza l'API ProjectX per operare su futures (MNQ, MGC, MYM).

## 🎯 Caratteristiche

- ✅ Trading completamente automatico
- ✅ Strategia Trend Following con EMA, RSI e ATR
- ✅ Backtesting completo con report dettagliati
- ✅ Risk management integrato con limiti di perdita
- ✅ Supporto multi-strumento (MNQ, MGC, MYM)
- ✅ Logging completo e database SQLite
- ✅ Stop loss e take profit automatici
- ✅ Real-time data streaming via WebSocket

## 📋 Requisiti

- Python 3.12 o superiore
- Account ProjectX con API access ($14.50/mese per utenti TopStepX)
- Account TopStepX (opzionale, per live trading)

## 🚀 Installazione

### 1. Clona o scarica il progetto

```bash
cd progettost
```

### 2. Crea un ambiente virtuale (raccomandato)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura le credenziali

Copia il file `.env.template` in `.env`:

```bash
copy .env.template .env  # Windows
cp .env.template .env    # Linux/Mac
```

Modifica il file `.env` con le tue credenziali:

```env
PROJECT_X_USERNAME=tuo_username
PROJECT_X_API_KEY=tua_api_key
TRADING_MODE=demo
PROJECT_X_ENVIRONMENT=demo
```

## 📖 Ottenere le Credenziali API

1. Vai su [dashboard.projectx.com](https://dashboard.projectx.com)
2. Crea un account se non ne hai uno
3. Naviga in "Subscriptions" > "ProjectX API Access"
4. Sottoscrivi il servizio ($14.50/mese con sconto TopStepX)
5. Vai in Settings e genera la tua API key
6. **IMPORTANTE**: Salva la chiave immediatamente, non potrai recuperarla dopo!

## 🎮 Utilizzo

### Backtest (Consigliato per iniziare)

Prima di fare trading live, è **essenziale** testare la strategia su dati storici:

```bash
python run_backtest.py
```

Il backtest:
- Scarica 90 giorni di dati storici (configurabile)
- Simula il trading con la strategia
- Genera report dettagliati con grafici
- Salva i risultati in `backtest_results/`

### Live/Demo Trading

**IMPORTANTE**: Testa sempre su demo prima di andare live!

```bash
python run_live.py
```

Il bot:
- Si connette all'API ProjectX
- Monitora i mercati in tempo reale
- Esegue trade automaticamente secondo la strategia
- Gestisce stop loss e take profit
- Rispetta i limiti di rischio configurati
- Salva tutti i log in `logs/` e database

#### Per passare da Demo a Live:

Modifica il file `.env`:

```env
TRADING_MODE=live
PROJECT_X_ENVIRONMENT=topstepx
```

**⚠️ ATTENZIONE LIVE TRADING:**
- Sei 100% responsabile per tutti i trade
- I trade via API NON sono reversibili da TopStepX
- Devi eseguire il bot dal tuo dispositivo personale (NO VPS/VPN)
- TopStepX chiuderà il tuo account se usi VPS/VPN

## ⚙️ Configurazione

### Modificare la Strategia

Apri `src/config.py` per modificare i parametri:

```python
# Timeframe
timeframe: str = '5min'  # Cambia a '1min', '15min', etc.

# Parametri EMA
ema_fast: int = 20  # EMA veloce
ema_slow: int = 50  # EMA lenta

# Parametri RSI
rsi_period: int = 14
rsi_overbought: float = 70.0
rsi_oversold: float = 30.0

# Parametri ATR (per stop loss dinamici)
atr_period: int = 14
atr_stop_multiplier: float = 2.0  # Stop = 2x ATR
atr_target_multiplier: float = 4.0  # Target = 4x ATR (R/R 1:2)
```

### Risk Management

Configura i limiti di rischio in `src/config.py`:

```python
# Perdite massime
max_daily_loss: float = 300.0  # Perdita giornaliera massima
max_trade_loss: float = 100.0  # Perdita massima per trade
emergency_close_loss: float = 400.0  # Emergency stop

# Posizioni
max_position_size: int = 1  # Contratti per posizione
max_total_positions: int = 3  # Posizioni totali simultanee

# Orari di trading
trading_start_hour: int = 9  # Inizio (ET)
trading_end_hour: int = 15  # Fine (ET)
```

### Strumenti da Tradare

Modifica gli strumenti in `src/config.py`:

```python
symbols: List[str] = ['MNQ', 'MGC', 'MYM']
```

Strumenti disponibili:
- **MNQ** - Micro E-mini Nasdaq-100 ($2/punto)
- **MGC** - Micro Gold ($1/punto)
- **MYM** - Micro E-mini Dow ($0.50/punto)
- **MES** - Micro E-mini S&P 500 ($5/punto)

## 📊 Strategia di Trading

### Logica della Strategia

Il bot utilizza una strategia **Trend Following** multi-indicatore:

#### Entry Signals

**LONG (Acquisto):**
- EMA veloce (20) incrocia sopra EMA lenta (50) ✅
- RSI < 70 (non overbought) ✅
- ATR > soglia minima (volatilità sufficiente) ✅

**SHORT (Vendita):**
- EMA veloce incrocia sotto EMA lenta ✅
- RSI > 30 (non oversold) ✅
- ATR > soglia minima ✅

#### Exit Rules

- **Stop Loss**: Entry ± (2 × ATR)
- **Take Profit**: Entry ± (4 × ATR)
- **Risk/Reward**: 1:2 minimo
- **Crossover inverso**: Chiude la posizione se EMA si inverte
- **Fuori orario**: Chiude tutte le posizioni fuori dall'orario di trading

## 📁 Struttura del Progetto

```
progettost/
├── src/
│   ├── config.py           # Configurazione generale
│   ├── data_manager.py     # Gestione dati storici e real-time
│   ├── strategy.py         # Logica della strategia
│   ├── risk_manager.py     # Gestione del rischio
│   ├── trading_engine.py   # Engine per live trading
│   ├── backtester.py       # Engine per backtesting
│   └── monitor.py          # Logging e monitoring
├── data/                   # Database e cache dati
├── logs/                   # File di log
├── backtest_results/       # Report e grafici backtest
├── run_backtest.py         # Script per backtesting
├── run_live.py            # Script per live trading
├── requirements.txt        # Dipendenze Python
├── .env.template          # Template configurazione
└── README.md              # Questa documentazione
```

## 📈 Metriche di Performance

Il sistema traccia automaticamente:

- **Total Return**: Rendimento totale %
- **Win Rate**: Percentuale di trade vincenti
- **Profit Factor**: Profitto lordo / Perdita lorda
- **Sharpe Ratio**: Rendimento aggiustato per il rischio
- **Max Drawdown**: Massima perdita dal picco
- **Average Win/Loss**: Guadagno/perdita media per trade
- **Expectancy**: Profitto atteso per trade

## 🔒 Sicurezza

### Protezioni Integrate

- ✅ Stop loss automatico su ogni trade
- ✅ Limite di perdita giornaliera
- ✅ Limite massimo di posizioni
- ✅ Emergency stop a $400 di perdita
- ✅ Stop automatico dopo 3 loss consecutive
- ✅ Chiusura posizioni fuori orario
- ✅ Verifica margin prima di ogni trade

### Best Practices

1. **Inizia su DEMO**: Testa sempre su demo prima del live
2. **Backtesta prima**: Verifica la strategia su dati storici
3. **Inizia piccolo**: Usa 1 contratto alla volta inizialmente
4. **Monitora costantemente**: Controlla il bot regolarmente
5. **Non modificare durante trade**: Evita di cambiare config con posizioni aperte
6. **Tieni log**: Rivedi i log per ottimizzare la strategia
7. **Rispetta device policy**: SOLO da dispositivo personale

## 🐛 Troubleshooting

### Errore: "API key not found"

- Verifica che il file `.env` esista
- Controlla che `PROJECT_X_USERNAME` e `PROJECT_X_API_KEY` siano impostati
- Assicurati che le credenziali siano corrette

### Errore: "No data available"

- Verifica la connessione internet
- Controlla che l'account ProjectX sia attivo
- Verifica che gli strumenti siano disponibili

### Bot non apre posizioni

- Controlla che sia nell'orario di trading (9:00-15:00 ET)
- Verifica che non ci siano limiti di rischio attivi
- Controlla i log in `logs/` per dettagli
- Aumenta il `min_atr_threshold` se mercato poco volatile

### Errore di autenticazione

- Rigenera l'API key su dashboard.projectx.com
- Verifica che la subscription API sia attiva
- Controlla che `PROJECT_X_ENVIRONMENT` sia corretto (demo/topstepx)

## 📊 Ottimizzazione

### Migliorare la Strategia

1. **Analizza i backtest**:
   - Controlla win rate, profit factor, drawdown
   - Identifica pattern nei trade perdenti
   - Testa diversi parametri

2. **Modifica i parametri**:
   - Cambia periodi EMA (es: 10/30, 50/200)
   - Aggiusta soglie RSI
   - Modifica moltiplicatori ATR per stop/target

3. **Aggiungi filtri**:
   - Volume minimo
   - Trend più forti
   - Sessioni specifiche

4. **Multi-timeframe**:
   - Conferma trend su timeframe superiore
   - Entry su timeframe inferiore

## 📝 Log e Database

### File di Log

- `logs/trading_YYYYMMDD.log`: Log giornalieri
- Livelli: DEBUG, INFO, WARNING, ERROR
- Configurabili in `src/config.py`

### Database SQLite

Il bot salva automaticamente in `data/trading_bot.db`:
- Tutti i trade completati
- Eventi importanti (segnali, posizioni aperte/chiuse)
- Statistiche giornaliere

Query esempio:

```python
import sqlite3
conn = sqlite3.connect('data/trading_bot.db')
cursor = conn.cursor()

# Ultimi 10 trade
cursor.execute("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10")
trades = cursor.fetchall()
```

## 🆘 Supporto

### Risorse

- **ProjectX API Docs**: https://gateway.docs.projectx.com/
- **ProjectX Python SDK**: https://project-x-py.readthedocs.io/
- **TopStepX Help**: https://help.topstep.com/

### Contatti

- Problemi con API: Contatta ProjectX support via dashboard
- Problemi con account: Contatta TopStepX support
- **NOTA**: TopStepX NON fornisce supporto per il codice

## ⚖️ Disclaimer

**IMPORTANTE - LEGGERE ATTENTAMENTE:**

- Questo software è fornito "AS IS" senza garanzie di alcun tipo
- Il trading di futures comporta rischi sostanziali
- Puoi perdere più del tuo investimento iniziale
- I risultati passati non garantiscono risultati futuri
- Sei l'unico responsabile per tutte le decisioni di trading
- L'autore non si assume responsabilità per perdite o danni
- Utilizza questo software solo se comprendi completamente i rischi
- Testa sempre su DEMO prima del live trading
- NON usare capitale che non puoi permetterti di perdere

**TopStepX Policy:**
- Trade via API non sono reversibili
- Devi usare SOLO il tuo dispositivo personale (no VPS/VPN)
- Violazioni delle policy = account termination
- Leggi i Terms of Service di TopStepX

## 📄 Licenza

Questo progetto è fornito per scopi educativi. Usalo a tuo rischio e pericolo.

## 🎓 Per Iniziare

### Workflow Consigliato

1. **Setup** (15 min)
   ```bash
   pip install -r requirements.txt
   cp .env.template .env
   # Modifica .env con le tue credenziali
   ```

2. **Primo Backtest** (10 min)
   ```bash
   python run_backtest.py
   ```
   - Analizza i risultati
   - Controlla grafici in `backtest_results/`

3. **Ottimizzazione** (variabile)
   - Modifica parametri in `src/config.py`
   - Esegui nuovi backtest
   - Confronta risultati

4. **Demo Trading** (1-7 giorni)
   ```bash
   # Assicurati: TRADING_MODE=demo nel .env
   python run_live.py
   ```
   - Monitora attentamente
   - Verifica che tutto funzioni
   - Controlla log e database

5. **Live Trading** (quando sei pronto)
   - Modifica `.env` a `TRADING_MODE=live`
   - Inizia con 1 contratto
   - Scala gradualmente
   - Monitora quotidianamente

## ✅ Checklist Pre-Live

Prima di andare live, assicurati di:

- [ ] Hai backtestato la strategia con buoni risultati
- [ ] Hai testato su demo per almeno 3-5 giorni
- [ ] Comprendi perfettamente come funziona la strategia
- [ ] Hai configurato correttamente i limiti di rischio
- [ ] Hai verificato che il bot gestisca gli errori
- [ ] Stai usando il tuo dispositivo personale (no VPS/VPN)
- [ ] Hai letto e compreso i Terms of Service TopStepX
- [ ] Sei preparato a perdere il capitale rischiato
- [ ] Puoi monitorare il bot durante le ore di trading

---

**Buon Trading! 🚀**

*Remember: Trade responsibly and never risk more than you can afford to lose.*
