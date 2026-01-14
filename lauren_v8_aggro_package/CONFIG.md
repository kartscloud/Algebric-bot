# LAUREN v8 AGGRO - Configuration Reference

## Position Sizing

```python
MAX_POSITIONS = 1                  # Single position - concentrate capital
MAX_POSITION_PCT = 0.50            # 50% of capital per trade
MAX_LOSS_PER_TRADE = 500.0         # Cap worst-case at $500
MAX_DAILY_LOSS = 800.0             # Stop if down $800
MAX_DAILY_PROFIT = 1500.0          # Lock in gains at $1500
MIN_BUYING_POWER_RESERVE = 50.0    # Keep $50 reserve
```

### How Position Size is Calculated

```python
# Whichever is LARGER (aggressive mode):
max_by_risk = $500 / (entry_price × 100)
max_by_pct = (capital - $50) × 50% / (entry_price × 100)

contracts = max(max_by_risk, max_by_pct)

# But never more than you can afford
contracts = min(contracts, affordable)
```

---

## Signal Thresholds

```python
LONG_THRESH = -0.30     # Score < -0.30 = underpriced = BUY
SHORT_THRESH = 0.40     # Score > 0.40 = overpriced = SELL
MIN_CONFIDENCE = 50     # Minimum adjusted confidence to trade
```

---

## Option Filters

```python
DTE_MIN = 5             # No 0DTE gamma bombs
DTE_MAX = 14            # Max 2 weeks out
MIN_OTM = 0.05          # At least 5% OTM
MAX_OTM = 0.20          # Max 20% OTM
MIN_PREMIUM = 0.10      # Min $0.10 option price
MIN_BID = 0.05          # Min $0.05 bid
MIN_OI = 25             # Min open interest
MAX_SPREAD_PCT = 0.15   # Max 15% bid-ask spread
```

---

## Exit Rules

```python
TARGET_PCT = 50         # Take profit at +50%
STOP_PCT = -40          # Stop loss at -40%
TIME_STOP_DTE = 1       # Exit when 1 day to expiry
```

### Exit Fill Model (Bid-Anchored)

```python
# TARGET: Can be patient, 55% toward mid
exit = bid + (mid - bid) × 0.55

# STOP: Need to get out, 15% toward mid
exit = bid + (mid - bid) × 0.15

# TIME: Somewhat urgent, 35% toward mid
exit = bid + (mid - bid) × 0.35
```

---

## Circuit Breakers

```python
MARKET_OPEN_DELAY_MINUTES = 5      # Wait 5 min after open
MAX_CONSECUTIVE_REJECTS = 5        # Stop after 5 rejected orders
MAX_SPREAD_BLOWOUT_PCT = 0.30      # Pause if spread > 30%
MIN_QUOTE_FRESHNESS_SEC = 120      # Quotes must be < 2 min old
```

---

## Entry Fill Model (Slippage)

```python
# Factors that increase slippage:
# - Higher confidence → more aggressive entry
# - Lower OI → harder to fill
# - Shorter DTE → more urgency
# - Wider spread → more uncertainty

slippage = (confidence_factor + oi_factor + time_factor + spread_factor) × spread × 0.5
entry = mid + slippage  # But never exceed ask
```

---

## Tuning Guide

### More Aggressive
```python
MAX_POSITION_PCT = 0.65           # 65% per trade
MAX_LOSS_PER_TRADE = 750.0        # Higher risk cap
MIN_OI = 15                        # Accept lower liquidity
MAX_SPREAD_PCT = 0.18             # Accept wider spreads
```

### More Conservative
```python
MAX_POSITION_PCT = 0.35           # 35% per trade
MAX_LOSS_PER_TRADE = 300.0        # Lower risk cap
MIN_OI = 50                        # Require better liquidity
MAX_SPREAD_PCT = 0.12             # Require tighter spreads
```

### Faster Exits
```python
TARGET_PCT = 35                    # Take profit earlier
STOP_PCT = -25                     # Cut losses faster
```

### More Patience
```python
TARGET_PCT = 75                    # Let winners run
STOP_PCT = -50                     # Give more room
```
