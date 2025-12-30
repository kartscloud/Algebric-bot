# DUAL TRADING SYSTEM

Two autonomous trading bots that split your buying power 50/50.

## System 1: QUANT TRADER (50% of buying power)
- Trades: SOXL, TQQQ, AMD, PLTR, SOFI, NIO, etc.
- Strategy: Multi-signal quant model (Graph Laplacian, HMM regime, momentum)
- Scans every 60 seconds

## System 2: LAUREN v6 (50% of buying power)  
- Trades: NVDL, TSLL (leveraged NVIDIA/Tesla ETFs)
- Strategy: Black-Scholes mispricing + macro regime
- Scans every 3 minutes
- Evolved from v5 signal bot → now fully autonomous

---

## HOW TO RUN

### Option A: Run Both at Once
```bash
./run_both.sh
```

### Option B: Run Separately (two terminals)

**Terminal 1:**
```bash
cd quant_trader
python main.py
```

**Terminal 2:**
```bash
cd lauren_auto
python lauren.py
```

---

## SETUP

### 1. Install packages
```bash
pip install robin_stocks numpy scipy
```

### 2. Run
```bash
./run_both.sh
```

Or run each one separately in different terminal windows.

---

## BUYING POWER SPLIT

| System | Share | With $1000 |
|--------|-------|------------|
| Quant Trader | 50% | $500 |
| LAUREN | 50% | $500 |

They won't interfere because:
1. They trade DIFFERENT tickers
2. They each only use 50% of buying power
3. They have separate position tracking

---

## FILES

```
trading_system/
├── run_both.sh          # Runs both systems
├── quant_trader/        # System 1
│   ├── main.py          # Run this
│   ├── config.py        # Settings
│   └── ...
└── lauren_auto/         # System 2
    └── lauren.py        # Run this
```

---

## WHAT YOU'LL SEE

**Quant Trader:**
```
2024-01-15 09:35:01 | INFO | CYCLE: 09:35:01
2024-01-15 09:35:03 | INFO | 🎯 OPENING POSITION
2024-01-15 09:35:03 | INFO |    SOXL 28.0 call
```

**LAUREN:**
```
2024-01-15 09:35:01 | LAUREN | INFO | CYCLE: 09:35:01
2024-01-15 09:35:03 | LAUREN | INFO | 🎯 SIGNAL: NVDL $95C
```

---

## TO STOP

Press `Ctrl + C` in each terminal.

---

## DAILY LIMITS

| System | Profit Target | Loss Limit |
|--------|---------------|------------|
| Quant Trader | +$50 | -$100 |
| LAUREN | +$25 | -$50 |

When a system hits its limit, it stops for the day.

---

## THAT'S IT

Run `./run_both.sh` and leave it open during market hours.
