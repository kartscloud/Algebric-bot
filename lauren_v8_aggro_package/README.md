# LAUREN v8 AGGRO

## Realistic Fills + Aggressive Sizing

**Backtest (December 2025):** 4 trades, 100% win rate, **+290% return**

This is the production-ready version that balances aggressive returns with survivable risk.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit credentials in lauren_v8_aggro.py (lines 37-38)
RH_USERNAME = "your_email@example.com"
RH_PASSWORD = "your_password"

# 3. Run
python lauren_v8_aggro.py

# Or use the shell script
chmod +x run.sh
./run.sh
```

---

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Position Size | 50% of capital | Aggressive but not suicidal |
| Max Loss/Trade | $500 | Worst case per position |
| Max Daily Loss | $800 | Stop trading for day |
| Max Daily Profit | $1,500 | Lock in gains |
| Target | +50% | Take profit |
| Stop | -40% | Cut losses |

### Signal Settings (Same as v7)
| Setting | Value |
|---------|-------|
| LONG Threshold | < -0.30 |
| SHORT Threshold | > 0.40 |
| Min Confidence | 50% |
| DTE Range | 5-14 days |
| OTM Range | 5-20% |

---

## What Makes v8 AGGRO Different

### vs v7 AGGRO (Fantasy Mode)
- ❌ v7: Fills at mid price (doesn't happen)
- ✅ v8A: Realistic entry slippage modeled
- ❌ v7: Exits at mid price (really doesn't happen)
- ✅ v8A: Exits anchored to bid (where you actually sell)
- ❌ v7: 100% capital per trade (one loss = game over)
- ✅ v8A: 50% capital, $500 max loss (survives losing streaks)

### vs v8 CONSERVATIVE
- v8C: $200 max loss → +60% return (safe but slow)
- v8A: $500 max loss → +290% return (aggressive but survivable)

---

## Backtest Results

| Trade | Direction | Entry | Exit | Contracts | P&L |
|-------|-----------|-------|------|-----------|-----|
| 1 | SHORT NVDL $98 | $1.65 | $0.56 | 3 | +$327 |
| 2 | SHORT NVDL $93 | $0.75 | $0.11 | 8 | +$512 |
| 3 | LONG NVDL $94 | $0.10 | $0.24 | 89 | +$1,246 |
| 4 | LONG NVDL $96 | $0.75 | $1.16 | 20 | +$820 |
| **TOTAL** | | | | | **+$2,905** |

**$1,000 → $3,905 (+290%)**

---

## Circuit Breakers

The bot will automatically pause when:

| Condition | Action |
|-----------|--------|
| First 5 min after open | Wait (chaos window) |
| Spread > 30% | Skip trade |
| Quote unchanged 2+ min | Skip (stale) |
| 5+ order rejects | Stop trading |
| Daily loss > $800 | Stop for day |
| Daily profit > $1,500 | Stop for day (lock gains) |

---

## Commands

```bash
# Run the bot
python lauren_v8_aggro.py

# View trade history
python lauren_v8_aggro.py --trades

# Clear state (fresh start)
python lauren_v8_aggro.py --clear
```

---

## Files Created

```
lauren_v8_aggro_data/
├── positions.json    # Current open positions
├── state.json        # Daily P&L tracking
├── trades.json       # Trade history
└── vol_cache.json    # Cached volatility

lauren_v8_aggro_logs/
└── lauren.log        # Execution log
```

---

## Risk Warning

- This is an **aggressive** trading bot
- Past backtest results don't guarantee future performance
- Only trade with money you can afford to lose
- Paper trade first to verify fills match expectations

---

## Recommended Progression

1. **Week 1-2:** Paper trade, verify fills
2. **Week 3-4:** Small real money ($500-1000)
3. **If win rate >70%:** Scale up gradually
4. **Never:** Go full v7 (100% sizing is gambling)
