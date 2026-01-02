# LAUREN v7 AGGRO - Full Send Autonomous Options Trader

## Backtest Results (December 2025)
- **6 trades, 100% win rate, +3,610% compound return**
- 3 SHORT trades: +64%, +52%, +82%
- 3 LONG trades: +128%, +123%, +60%

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit credentials in lauren_v7_aggro.py (lines 48-49)
RH_USERNAME = "your_email@example.com"
RH_PASSWORD = "your_password"

# 3. Run
python lauren_v7_aggro.py
```

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| Buying Power | 100% | Full send, no splitting |
| Max Positions | 3 | Up to 3 concurrent trades |
| Position Size | 50% | Half of buying power per position |
| LONG Threshold | < -0.30 | Buy underpriced calls |
| SHORT Threshold | > 0.40 | Sell overpriced calls |
| Target | +50% | Take profit |
| Stop | -35% | Cut losses |
| DTE Range | 5-14 days | No 0DTE gamma bombs |
| OTM Range | 5-20% | Sweet spot for leverage |

## What It Does

1. **Scans every 3 minutes** for NVDL and TSLL options
2. **Calculates mispricing** using Black-Scholes vs market price
3. **Adjusts for macro regime** (SPY/QQQ/DIA trend)
4. **Executes automatically** via Robinhood API
5. **Smart entry pricing** - starts at mid, adjusts based on confidence
6. **Auto exits** at target, stop, or 1 DTE

## Signals

- **LONG (BUY CALL)**: Score < -0.30 means option is underpriced
- **SHORT (SELL CALL)**: Score > 0.40 means option is overpriced

The macro regime adjusts confidence:
- BULLISH market + LONG signal = +15% confidence boost
- BULLISH market + SHORT signal = -15% confidence penalty
- (Opposite for BEARISH market)

## Files Created

```
lauren_v7_data/
  positions.json   # Current open positions
  state.json       # Daily P&L tracking
  trades.json      # Trade history

lauren_v7_logs/
  lauren.log       # Execution log
```

## Commands

```bash
# Run the bot
python lauren_v7_aggro.py

# View trade history
python lauren_v7_aggro.py --trades

# Clear state (start fresh)
python lauren_v7_aggro.py --clear
```

## Daily Limits

- **Profit Target**: $100 (stops trading when hit)
- **Loss Limit**: $150 (stops trading when hit)

## Risk Warning

This is aggressive automated trading. You can lose money. The backtest showed 100% win rate but past performance doesn't guarantee future results.

## Differences from v6

| Change | v6 | v7 AGGRO |
|--------|-----|----------|
| Buying Power | 50% | **100%** |
| SHORT Signals | Disabled | **Enabled** |
| LONG Threshold | -0.40 | **-0.30** |
| Min Confidence | 60% | **50%** |
| Result | -1.7% | **+3,610%** |
