#!/usr/bin/env python3
"""
================================================================================
QUANTITATIVE OPTIONS TRADING SYSTEM - COMPLETE IMPLEMENTATION
================================================================================

This is a comprehensive quant trading system integrating:

1. BLACK-SCHOLES MISPRICING DETECTION (LAUREN-style)
   - Calculate theoretical option prices
   - Compare to market prices
   - Find underpriced/overpriced options

2. GRAPH LAPLACIAN SIGNAL
   - Build correlation network of underlyings
   - Apply diffusion: h = (I - αL)^J x
   - Residual e = x - h identifies mispriced assets

3. HIDDEN MARKOV MODEL REGIME DETECTION
   - 3-state model: Low Vol / Normal / High Vol
   - Adjust position sizing based on regime
   - Reduce exposure in crisis regimes

4. MULTI-FACTOR ALPHA MODEL
   - Momentum (multiple timeframes)
   - Mean reversion
   - Volatility signals
   - IV percentile

5. RISK MANAGEMENT
   - Position sizing based on Kelly criterion
   - Portfolio Greeks limits
   - Stop losses and profit targets
   - Daily P&L limits

6. LIVE EXECUTION
   - Robinhood integration
   - Minute-level polling
   - Automatic order execution

================================================================================
CONFIGURATION
================================================================================
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

# =============================================================================
# AUTHENTICATION - YOUR CREDENTIALS
# =============================================================================

ROBINHOOD_USERNAME = "kartikaygeorg@gmail.com"
ROBINHOOD_PASSWORD = "SUPERsuace101!"

# =============================================================================
# TRADING MODE
# =============================================================================

PAPER_TRADING = False  # <<< LIVE TRADING ENABLED

# =============================================================================
# ACCOUNT CONFIGURATION
# =============================================================================

ACCOUNT_SIZE = 1000.0          # Starting capital
DAILY_PROFIT_TARGET = 50.0     # Stop when up $50 (5%)
DAILY_LOSS_LIMIT = 100.0       # Stop when down $100 (10%)

# =============================================================================
# POSITION SIZING
# =============================================================================

MAX_POSITION_PCT = 0.35        # 35% per position ($350 max)
MAX_POSITIONS = 3              # Maximum concurrent positions
MAX_EXPOSURE_PCT = 0.95        # 95% max capital deployed
MIN_TRADE_SIZE = 25.0          # Minimum $25 per trade

# =============================================================================
# OPTION SELECTION FILTERS
# =============================================================================

# Expiration
MIN_DTE = 0                    # Same-day expiration OK
MAX_DTE = 14                   # Max 2 weeks out

# Premium (cheap options for small account)
MIN_PREMIUM = 0.05             # $5 minimum per contract
MAX_PREMIUM = 2.50             # $250 maximum per contract

# Greeks
MIN_DELTA = 0.25               # Not too far OTM
MAX_DELTA = 0.75               # Not too deep ITM

# Liquidity
MIN_OPEN_INTEREST = 200        # Minimum open interest
MIN_VOLUME = 50                # Minimum daily volume
MAX_SPREAD_PCT = 0.10          # Max 10% bid-ask spread

# Implied Volatility
MIN_IV = 0.20                  # At least 20% IV
MAX_IV = 2.50                  # Cap extreme IV

# =============================================================================
# EXIT RULES
# =============================================================================

# Profit Taking
SCALP_TARGET_PCT = 0.12        # 12% quick scalp (after 20 min)
PROFIT_TARGET_PCT = 0.40       # 40% main profit target
BIG_WIN_TARGET_PCT = 0.75      # 75% let winners run

# Stop Losses
STOP_LOSS_PCT = 0.30           # 30% hard stop
TRAILING_STOP_PCT = 0.18       # 18% trailing from high
TIGHT_STOP_PCT = 0.15          # 15% tight stop for scalps

# Time-based
TIME_STOP_MINUTES = 90         # Exit flat positions after 90 min
MIN_HOLD_FOR_SCALP = 20        # Hold at least 20 min before scalping

# =============================================================================
# SIGNAL CONFIGURATION
# =============================================================================

# Mispricing Signal (Black-Scholes)
MISPRICING_WEIGHT = 0.35       # 35% weight
MIN_MISPRICING_PCT = 0.025     # 2.5% mispricing threshold

# IV Percentile Signal
IV_WEIGHT = 0.20               # 20% weight
IV_BUY_THRESHOLD = 0.30        # Buy when IV < 30th percentile
IV_SELL_THRESHOLD = 0.70       # Sell when IV > 70th percentile
IV_LOOKBACK_MINUTES = 500      # ~8 hours of minute data

# Momentum Signal
MOMENTUM_WEIGHT = 0.25         # 25% weight
MOMENTUM_LOOKBACK = 15         # 15-minute momentum
MOMENTUM_THRESHOLD = 0.003     # 0.3% move threshold

# Graph Signal (Laplacian residual)
GRAPH_WEIGHT = 0.10            # 10% weight
CORRELATION_WINDOW = 20        # 20 periods for correlation
DIFFUSION_ALPHA = 0.3          # Diffusion strength
DIFFUSION_STEPS = 2            # Number of diffusion iterations

# Regime Signal
REGIME_WEIGHT = 0.10           # 10% weight
REGIME_LOOKBACK = 50           # Periods for regime detection

# Minimum signal to trade
MIN_SIGNAL_SCORE = 0.20        # Minimum composite score

# =============================================================================
# REGIME-BASED ADJUSTMENTS
# =============================================================================

REGIME_POSITION_MULTIPLIER = {
    0: 1.3,    # Low volatility - increase size
    1: 1.0,    # Normal - baseline
    2: 0.5,    # High volatility - reduce size
}

REGIME_STOP_MULTIPLIER = {
    0: 1.2,    # Low vol - wider stops
    1: 1.0,    # Normal
    2: 0.7,    # High vol - tighter stops
}

# =============================================================================
# UNDERLYING UNIVERSE
# =============================================================================

# Primary targets - cheap, liquid options
PRIMARY_SYMBOLS = [
    # Leveraged ETFs (big moves, cheap premiums)
    'SOXL',    # 3x Semiconductors
    'TQQQ',    # 3x Nasdaq
    'UPRO',    # 3x S&P 500
    'TNA',     # 3x Small Cap
    'LABU',    # 3x Biotech
    
    # Volatile tech stocks
    'AMD',     # Advanced Micro Devices
    'NVDA',    # NVIDIA (liquid)
    'PLTR',    # Palantir (~$25)
    'SOFI',    # SoFi (~$12)
    'HOOD',    # Robinhood (~$25)
    
    # Cheap stocks with options
    'NIO',     # NIO (~$5)
    'RIVN',    # Rivian (~$12)
    'LCID',    # Lucid (~$3)
    'F',       # Ford (~$10)
    'SNAP',    # Snap (~$12)
    
    # Meme stocks (high IV)
    'GME',     # GameStop
    'AMC',     # AMC
]

# Secondary targets (if primary not available)
SECONDARY_SYMBOLS = [
    'MARA',    # Marathon Digital
    'RIOT',    # Riot Platforms
    'COIN',    # Coinbase
    'SQ',      # Block
    'PYPL',    # PayPal
    'UBER',    # Uber
    'LYFT',    # Lyft
    'DKNG',    # DraftKings
]

# All symbols combined
ALL_SYMBOLS = PRIMARY_SYMBOLS + SECONDARY_SYMBOLS

# =============================================================================
# TIMING CONFIGURATION
# =============================================================================

POLL_INTERVAL_SECONDS = 60     # Check every minute
AVOID_FIRST_MINUTES = 3        # Skip first 3 minutes
AVOID_LAST_MINUTES = 5         # Skip last 5 minutes

# Market hours (Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# =============================================================================
# LOGGING AND DATA
# =============================================================================

LOG_LEVEL = "INFO"
DATA_DIR = "data"
LOG_DIR = "logs"
STATE_FILE = "state.json"
TRADE_LOG_FILE = "trades.json"

# =============================================================================
# ENUMS
# =============================================================================

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class SignalType(Enum):
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2

class Regime(Enum):
    LOW_VOL = 0
    NORMAL = 1
    HIGH_VOL = 2

class ExitReason(Enum):
    SCALP = "SCALP"
    PROFIT_TARGET = "PROFIT_TARGET"
    BIG_WIN = "BIG_WIN"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TIME_STOP = "TIME_STOP"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    MANUAL = "MANUAL"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptionQuote:
    """Complete option quote data"""
    underlying: str
    underlying_price: float
    strike: float
    expiration: str
    option_type: OptionType
    
    # Prices
    bid: float
    ask: float
    mark: float
    last: float
    
    # Volume
    volume: int
    open_interest: int
    
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        if self.mark <= 0:
            return 999.0
        return self.spread / self.mark
    
    @property
    def dte(self) -> int:
        exp = datetime.strptime(self.expiration, '%Y-%m-%d')
        return max(0, (exp.date() - datetime.now().date()).days)
    
    @property
    def moneyness(self) -> float:
        if self.option_type == OptionType.CALL:
            return self.underlying_price / self.strike
        return self.strike / self.underlying_price
    
    @property
    def is_itm(self) -> bool:
        if self.option_type == OptionType.CALL:
            return self.underlying_price > self.strike
        return self.underlying_price < self.strike


@dataclass
class Position:
    """Tracked position with full details"""
    # Identification
    position_id: str
    underlying: str
    strike: float
    expiration: str
    option_type: str
    
    # Size
    contracts: int
    entry_price: float
    
    # Timing
    entry_time: datetime
    entry_signal_score: float
    entry_regime: int
    
    # Tracking
    current_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 999.0
    
    # Greeks at entry
    entry_delta: float = 0.0
    entry_gamma: float = 0.0
    entry_iv: float = 0.0
    
    def __post_init__(self):
        self.high_price = self.entry_price
        self.low_price = self.entry_price
        self.current_price = self.entry_price
        
    def update_price(self, price: float):
        """Update current price and track high/low"""
        self.current_price = price
        if price > self.high_price:
            self.high_price = price
        if price < self.low_price:
            self.low_price = price
            
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.contracts * 100
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price
    
    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.contracts * 100
    
    @property
    def market_value(self) -> float:
        return self.current_price * self.contracts * 100
    
    @property
    def hold_time_minutes(self) -> float:
        return (datetime.now() - self.entry_time).total_seconds() / 60
    
    @property
    def drawdown_from_high(self) -> float:
        if self.high_price <= 0:
            return 0.0
        return (self.current_price - self.high_price) / self.high_price


@dataclass
class Signal:
    """Trading signal with all components"""
    # Option info
    underlying: str
    strike: float
    expiration: str
    option_type: OptionType
    
    # Signal scores (each -1 to +1)
    mispricing_score: float = 0.0
    iv_score: float = 0.0
    momentum_score: float = 0.0
    graph_score: float = 0.0
    regime_score: float = 0.0
    
    # Composite
    composite_score: float = 0.0
    signal_type: SignalType = SignalType.HOLD
    confidence: float = 0.0
    
    # Supporting data
    theoretical_price: float = 0.0
    market_price: float = 0.0
    implied_volatility: float = 0.0
    iv_percentile: float = 0.5
    current_regime: int = 1
    
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_composite(self):
        """Calculate weighted composite score"""
        self.composite_score = (
            MISPRICING_WEIGHT * self.mispricing_score +
            IV_WEIGHT * self.iv_score +
            MOMENTUM_WEIGHT * self.momentum_score +
            GRAPH_WEIGHT * self.graph_score +
            REGIME_WEIGHT * self.regime_score
        )
        
        # Determine signal type
        if self.composite_score >= 0.5:
            self.signal_type = SignalType.STRONG_BUY
        elif self.composite_score >= 0.2:
            self.signal_type = SignalType.BUY
        elif self.composite_score <= -0.5:
            self.signal_type = SignalType.STRONG_SELL
        elif self.composite_score <= -0.2:
            self.signal_type = SignalType.SELL
        else:
            self.signal_type = SignalType.HOLD
            
        # Calculate confidence (agreement of signals)
        scores = [self.mispricing_score, self.iv_score, 
                  self.momentum_score, self.graph_score]
        agreement = sum(1 for s in scores if s * self.composite_score > 0)
        self.confidence = (agreement / len(scores)) * abs(self.composite_score)


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    starting_balance: float
    current_balance: float = 0.0
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Trades
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Best/Worst
    biggest_win: float = 0.0
    biggest_loss: float = 0.0
    
    # Signals
    signals_generated: int = 0
    signals_traded: int = 0
    
    def __post_init__(self):
        self.current_balance = self.starting_balance
        
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def profit_factor(self) -> float:
        if self.biggest_loss == 0:
            return float('inf') if self.biggest_win > 0 else 0.0
        return abs(self.biggest_win / self.biggest_loss)
    
    def record_trade(self, pnl: float):
        """Record a completed trade"""
        self.total_trades += 1
        self.realized_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            if pnl > self.biggest_win:
                self.biggest_win = pnl
        else:
            self.losing_trades += 1
            if pnl < self.biggest_loss:
                self.biggest_loss = pnl


# Print configuration on import
def print_config():
    """Print current configuration"""
    mode = "🔴 LIVE TRADING" if not PAPER_TRADING else "🟢 PAPER TRADING"
    print(f"""
{'='*70}
QUANTITATIVE OPTIONS TRADING SYSTEM
{'='*70}
Mode:           {mode}
Account:        ${ACCOUNT_SIZE:,.0f}
Max Position:   ${ACCOUNT_SIZE * MAX_POSITION_PCT:,.0f} ({MAX_POSITION_PCT:.0%})
Max Positions:  {MAX_POSITIONS}
Daily Target:   +${DAILY_PROFIT_TARGET:,.0f}
Daily Stop:     -${DAILY_LOSS_LIMIT:,.0f}
{'='*70}
Symbols:        {len(ALL_SYMBOLS)} ({', '.join(PRIMARY_SYMBOLS[:5])}...)
DTE Range:      {MIN_DTE}-{MAX_DTE} days
Premium Range:  ${MIN_PREMIUM:.2f}-${MAX_PREMIUM:.2f}
Delta Range:    {MIN_DELTA:.2f}-{MAX_DELTA:.2f}
{'='*70}
Exit Rules:
  Scalp:        +{SCALP_TARGET_PCT:.0%} (after {MIN_HOLD_FOR_SCALP} min)
  Profit:       +{PROFIT_TARGET_PCT:.0%}
  Big Win:      +{BIG_WIN_TARGET_PCT:.0%}
  Stop Loss:    -{STOP_LOSS_PCT:.0%}
  Trailing:     -{TRAILING_STOP_PCT:.0%} from high
  Time Stop:    {TIME_STOP_MINUTES} min flat
{'='*70}
Signal Weights:
  Mispricing:   {MISPRICING_WEIGHT:.0%}
  IV Signal:    {IV_WEIGHT:.0%}
  Momentum:     {MOMENTUM_WEIGHT:.0%}
  Graph:        {GRAPH_WEIGHT:.0%}
  Regime:       {REGIME_WEIGHT:.0%}
{'='*70}
    """)
