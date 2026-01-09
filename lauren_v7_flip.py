#!/usr/bin/env python3
"""
================================================================================
LAUREN v7 FLIP - ACCOUNT FLIPPER EDITION
================================================================================

🎯 GOAL: Flip small accounts with disciplined risk management

STRATEGY:
1. PATIENCE: Wait for quality signals, don't chase
2. LET WINNERS RIDE: Trailing stop locks in gains
3. CUT LOSERS FAST: Strict -35% stop, no hope trades
4. FULL SEND: 100% on single best signal

EXIT RULES:
┌─────────────────────────────────────────────────────────────────────┐
│ STOP LOSS:      -35% (HARD CUT - no exceptions)                     │
│ TRAILING STOP:  Activates at +40%, trails by 25%                    │
│                 If +40% → trail stop at +15%                        │
│                 If +80% → trail stop at +55%                        │
│                 If +100%→ trail stop at +75%                        │
│ TIME STOP:      Exit at 2 DTE (theta burn protection)               │
│ PROFIT LOCK:    At +100%, lock in +75% minimum                      │
└─────────────────────────────────────────────────────────────────────┘

ANTI-DUMB-TRADE FILTERS:
✓ Score strength filter (no weak signals)
✓ IV spike filter (no chasing after move)
✓ Time of day filter (skip open/close)
✓ Volume/OI filter (ensure liquidity)
✓ Spread filter (ensure good fills)
✓ Daily trade limit (no overtrading)
✓ Cooldown after exit (no revenge trades)

================================================================================
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from math import log, sqrt, exp
import logging
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Robinhood credentials (same as quant_trader)
RH_USERNAME = "kartikaygeorg@gmail.com"
RH_PASSWORD = "SUPERsuace101!"

# Tickers LAUREN focuses on (different from quant_trader)
TICKERS = ['NVDL', 'TSLL']

# Historical volatility - FALLBACK VALUES (dynamic calc preferred)
HIST_VOL_FALLBACK = {
    'NVDL': 0.65,  # 2x NVDA
    'TSLL': 0.90,  # 2x TSLA
}

# Underlying tickers for vol calculation
UNDERLYING = {
    'NVDL': 'NVDA',
    'TSLL': 'TSLA'
}

# Leverage multipliers
LEVERAGE = {
    'NVDL': 2.0,
    'TSLL': 2.0
}

# Vol bounds
VOL_MIN = 0.30
VOL_MAX = 2.00
VOL_LOOKBACK_DAYS = 20

# HV MODE: 'dynamic' (calculates from last 20 days) or 'static' (uses fallback)
# Dynamic: More accurate, fewer false signals, but may miss some trades
# Static: More signals, but may generate false signals when vol is actually low
HV_MODE = 'static'  # Changed back to static - dynamic was too conservative

# Map to underlying for news
NEWS_TICKERS = {
    'NVDL': 'NVDA',
    'TSLL': 'TSLA'
}

# =============================================================================
# BUYING POWER - FULL SEND FOR ACCOUNT FLIP
# =============================================================================

BUYING_POWER_SHARE = 1.00  # 100% - FULL SEND for account flip
MAX_POSITION_PCT = 1.00    # 100% per position (concentrated)
MAX_POSITIONS = 1          # SINGLE POSITION - all in on best signal

# =============================================================================
# PATIENCE SETTINGS - DON'T CHASE, WAIT FOR QUALITY
# =============================================================================

SCAN_INTERVAL = 180        # 3 minutes between scans
MAX_TRADES_PER_DAY = 2     # Don't overtrade - quality over quantity
MIN_SCORE_STRENGTH = 0.15  # Score must be at least 0.15 away from threshold
                           # LONG needs score < -0.30, so requires < -0.45
                           # SHORT needs score > 0.40, so requires > 0.55
                           # This filters out borderline signals

# =============================================================================
# SIGNAL THRESHOLDS - STRICT FOR QUALITY
# =============================================================================

SHORT_THRESH = 0.40  # Score > 0.40 = overpriced = SHORT
LONG_THRESH = -0.30  # Score < -0.30 = underpriced = LONG

# =============================================================================
# ANTI-CHASE FILTERS - DON'T BUY LATE
# =============================================================================

MAX_IV_SPIKE = 1.50        # Don't buy if IV > 150% of HV (already priced in)
MAX_INTRADAY_MOVE = 0.30   # Don't buy if option already moved 30% today
MIN_TIME_TO_CLOSE = 30     # Don't enter within 30 min of market close
AVOID_FIRST_15_MIN = True  # Skip first 15 min (wild spreads)

# =============================================================================
# OPTION QUALITY FILTERS
# =============================================================================

DTE_MIN = 5          # Min days to expiry (no 0DTE gamma bombs)
DTE_MAX = 10         # Max 10 days (tighter = better gamma)
MIN_OTM = 0.02       # Min 2% OTM
MAX_OTM = 0.12       # Max 12% OTM (tighter = better delta)
MIN_PREMIUM = 0.10   # Min $0.10 option price
MIN_BID = 0.05       # Min $0.05 bid
MIN_OI = 50          # Min 50 open interest (better liquidity)
MIN_VOLUME = 50      # Min 50 volume today (active contract)
MAX_SPREAD_PCT = 0.15  # Max 15% spread (reasonable fills)

# =============================================================================
# EXIT RULES - LET WINNERS RIDE, CUT LOSERS FAST
# =============================================================================

# STOP LOSS - STRICT, NO EXCEPTIONS
STOP = -35                  # -35% hard stop (cut losers fast!)

# TRAILING STOP - LET WINNERS RIDE
TRAILING_STOP = True        # Enable trailing stop
TRAIL_ACTIVATE = 40         # Activate trailing at +40%
TRAIL_DISTANCE = 25         # Trail by 25% (if at +40%, stop at +15%)

# PROFIT LOCK - PROTECT BIG GAINS
PROFIT_LOCK_THRESH = 100    # At +100%, lock in gains
PROFIT_LOCK_MIN = 75        # Lock in at least +75%

# TIME STOP - THETA PROTECTION
TIME_STOP_DTE = 2           # Exit if 2 days to expiry

# =============================================================================
# CONFIDENCE - HIGHER BAR FOR QUALITY
# =============================================================================

MIN_CONFIDENCE = 60  # Min 60% confidence (raised from 50%)


# =============================================================================
# DAILY LIMITS - AGGRO
# =============================================================================

DAILY_PROFIT_TARGET = 100.0  # Stop when up $100 (more room to run)
DAILY_LOSS_LIMIT = 150.0     # Stop when down $150 (more risk tolerance)

# =============================================================================
# FILES
# =============================================================================

DATA_DIR = "lauren_v7_data"
LOG_DIR = "lauren_v7_logs"
POS_FILE = f"{DATA_DIR}/positions.json"
STATE_FILE = f"{DATA_DIR}/state.json"
TRADES_FILE = f"{DATA_DIR}/trades.json"

# =============================================================================
# LOGGING
# =============================================================================

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | LAUREN | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{LOG_DIR}/lauren.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# BLACK-SCHOLES
# =============================================================================

def bs_call(S: float, K: float, T: float, vol: float) -> float:
    """Black-Scholes call price"""
    if T <= 0 or vol <= 0:
        return 0
    d1 = (log(S/K) + (0.05 + 0.5*vol**2)*T) / (vol*sqrt(T))
    d2 = d1 - vol*sqrt(T)
    return S * norm.cdf(d1) - K * exp(-0.05*T) * norm.cdf(d2)


def calc_iv(price: float, S: float, K: float, T: float) -> Optional[float]:
    """Calculate implied volatility"""
    if price <= 0 or T <= 0:
        return None
    try:
        return brentq(lambda v: bs_call(S, K, T, v) - price, 0.01, 5.0)
    except:
        return None


# =============================================================================
# DYNAMIC VOLATILITY (from v8 - catches IV spikes better)
# =============================================================================

# Cache for volatility to avoid repeated API calls
_vol_cache = {}
_vol_cache_time = {}

def calculate_realized_vol(rh, ticker: str) -> float:
    """
    Calculate realized volatility from last 20 days of price data.
    This catches IV spikes better than static HV.
    
    If TSLL was calm recently → lower HV → same IV looks MORE overpriced → SHORT signal
    If TSLL was wild recently → higher HV → same IV looks normal → no signal
    """
    global _vol_cache, _vol_cache_time
    
    # Check cache (refresh every 30 min)
    cache_key = ticker
    now = datetime.now()
    if cache_key in _vol_cache:
        cache_age = (now - _vol_cache_time.get(cache_key, now)).total_seconds()
        if cache_age < 1800:  # 30 min cache
            return _vol_cache[cache_key]
    
    underlying = UNDERLYING.get(ticker, ticker)
    leverage = LEVERAGE.get(ticker, 1.0)
    
    try:
        historicals = rh.stocks.get_stock_historicals(
            underlying, interval='day', span='month', bounds='regular'
        )
        
        if not historicals or len(historicals) < 10:
            logger.debug(f"Not enough data for {ticker}, using fallback HV")
            return HIST_VOL_FALLBACK.get(ticker, 0.65)
        
        closes = [float(h['close_price']) for h in historicals if h.get('close_price')]
        if len(closes) < 10:
            return HIST_VOL_FALLBACK.get(ticker, 0.65)
        
        # Calculate realized vol
        returns = np.diff(np.log(closes))
        daily_vol = np.std(returns)
        annual_vol = daily_vol * sqrt(252)
        leveraged_vol = annual_vol * leverage
        
        # Clamp to reasonable range
        leveraged_vol = max(VOL_MIN, min(VOL_MAX, leveraged_vol))
        
        # Cache it
        _vol_cache[cache_key] = leveraged_vol
        _vol_cache_time[cache_key] = now
        
        logger.info(f"📊 Dynamic HV for {ticker}: {leveraged_vol:.0%} (from {len(closes)} days)")
        return leveraged_vol
        
    except Exception as e:
        logger.warning(f"Vol calc error for {ticker}: {e}, using fallback")
        return HIST_VOL_FALLBACK.get(ticker, 0.65)


def get_historical_vol(rh, ticker: str) -> float:
    """
    Get historical volatility - mode controlled by HV_MODE setting.
    
    Static mode: Uses HIST_VOL_FALLBACK values (0.90 for TSLL, 0.65 for NVDL)
    Dynamic mode: Calculates from last 20 days of price data
    
    Static tends to generate more LONG signals.
    Dynamic is more accurate but may miss some trades.
    """
    if HV_MODE == 'static':
        return HIST_VOL_FALLBACK.get(ticker, 0.65)
    
    if rh and hasattr(rh, 'stocks'):
        return calculate_realized_vol(rh, ticker)
    return HIST_VOL_FALLBACK.get(ticker, 0.65)


def quantum_score(price: float, S: float, K: float, T: float, hv: float) -> Tuple[Optional[float], Optional[float]]:
    """Calculate mispricing score"""
    iv = calc_iv(price, S, K, T)
    if not iv or iv < 0.1 or iv > 3:
        return None, None
    bs = bs_call(S, K, T, hv)
    if bs < 0.01:
        return None, None
    return 0.4 * (iv - hv) / hv + 0.6 * (price - bs) / bs, iv


def score_to_confidence(score: float) -> float:
    """Convert raw score to confidence (0-100)"""
    abs_score = abs(score)
    if abs_score <= 0.30:
        return 40 + (abs_score / 0.30) * 10
    elif abs_score <= 0.60:
        return 50 + ((abs_score - 0.30) / 0.30) * 25
    elif abs_score <= 1.0:
        return 75 + ((abs_score - 0.60) / 0.40) * 20
    return 95


def calculate_optimal_entry(
    desired_price: float,
    ask_price: float,
    bid_price: float,
    confidence: float,
    spot_price: float,
    strike: float,
    dte: int,
    volume: int,
    open_interest: int
) -> Tuple[float, str]:
    """
    Calculate optimal entry price based on multiple factors.
    
    Returns (optimal_price, reasoning)
    """
    spread = ask_price - bid_price
    spread_pct = spread / desired_price if desired_price > 0 else 0
    mid = (ask_price + bid_price) / 2
    
    # =========================================================================
    # FACTOR 1: Expected ROI
    # Calculate theoretical upside if underlying moves 3%
    # =========================================================================
    otm_pct = (strike - spot_price) / spot_price
    
    # Rough delta estimate based on OTM %
    if otm_pct <= 0:
        delta_est = 0.65  # ITM
    elif otm_pct < 0.05:
        delta_est = 0.50  # ATM
    elif otm_pct < 0.10:
        delta_est = 0.35  # Slightly OTM
    else:
        delta_est = 0.20  # OTM
    
    # Expected option move if underlying moves 3%
    underlying_move = spot_price * 0.03
    option_move = underlying_move * delta_est
    expected_roi = option_move / desired_price if desired_price > 0 else 0
    
    # =========================================================================
    # FACTOR 2: Liquidity Score (0-1)
    # Higher volume/OI = easier to get filled = less premium needed
    # =========================================================================
    vol_score = min(1.0, volume / 500) if volume else 0
    oi_score = min(1.0, open_interest / 1000) if open_interest else 0
    liquidity_score = (vol_score * 0.6 + oi_score * 0.4)
    
    # =========================================================================
    # FACTOR 3: Spread Score (0-1)
    # Tighter spread = more efficient market = stay closer to mid
    # =========================================================================
    if spread_pct < 0.03:
        spread_score = 1.0  # Very tight
    elif spread_pct < 0.06:
        spread_score = 0.8  # Tight
    elif spread_pct < 0.10:
        spread_score = 0.5  # Normal
    elif spread_pct < 0.15:
        spread_score = 0.3  # Wide
    else:
        spread_score = 0.1  # Very wide
    
    # =========================================================================
    # FACTOR 4: Time Pressure (0-1)
    # Shorter DTE = more urgent = willing to pay more
    # =========================================================================
    if dte <= 3:
        time_pressure = 0.9  # Very urgent
    elif dte <= 7:
        time_pressure = 0.6  # Urgent
    elif dte <= 14:
        time_pressure = 0.3  # Normal
    else:
        time_pressure = 0.1  # Plenty of time
    
    # =========================================================================
    # CALCULATE MAXIMUM PREMIUM WILLING TO PAY
    # =========================================================================
    
    # Base premium willingness from confidence (0-15%)
    # confidence 60 = 0%, confidence 100 = 15%
    conf_factor = max(0, (confidence - 60) / 40)  # 0 to 1
    base_premium = conf_factor * 0.15
    
    # Adjust by expected ROI - if ROI is high, willing to pay more
    # ROI of 50%+ = full premium, ROI of 10% = half premium
    roi_factor = min(1.0, expected_roi / 0.50)
    
    # Adjust by liquidity - low liquidity = need to pay more
    # High liquidity = can be patient
    liquidity_factor = 1.0 - (liquidity_score * 0.5)  # 0.5 to 1.0
    
    # Adjust by spread - wide spread = market is uncertain
    spread_factor = 1.0 - (spread_score * 0.3)  # 0.7 to 1.0
    
    # Adjust by time pressure
    time_factor = 1.0 + (time_pressure * 0.3)  # 1.0 to 1.3
    
    # Final premium calculation
    max_premium_pct = base_premium * roi_factor * liquidity_factor * spread_factor * time_factor
    
    # Cap at spread (never pay more than ask)
    max_premium_pct = min(max_premium_pct, spread_pct)
    
    # Calculate optimal price
    optimal_price = desired_price * (1 + max_premium_pct)
    optimal_price = min(optimal_price, ask_price)  # Never exceed ask
    optimal_price = round(optimal_price, 2)
    
    # Build reasoning string
    reasoning = (
        f"ROI:{expected_roi*100:.0f}% "
        f"Liq:{liquidity_score:.1f} "
        f"Sprd:{spread_score:.1f} "
        f"Time:{time_pressure:.1f} "
        f"→ +{max_premium_pct*100:.1f}%"
    )
    
    return optimal_price, reasoning


# =============================================================================
# MACRO SCANNER
# =============================================================================

def fetch_index_data(rh) -> Dict:
    """Fetch last 20 days of SPY, QQQ, DIA"""
    indices = {'SPY': 'SPY', 'QQQ': 'QQQ', 'DIA': 'DIA'}
    results = {}
    
    for name, ticker in indices.items():
        try:
            historicals = rh.stocks.get_stock_historicals(
                ticker, interval='day', span='month'
            )
            
            if historicals and len(historicals) >= 20:
                closes = [float(h['close_price']) for h in historicals[-20:]]
                ret_20d = (closes[-1] - closes[0]) / closes[0] * 100
                sma_20 = np.mean(closes)
                
                results[name] = {
                    'price': closes[-1],
                    'ret_20d': ret_20d,
                    'above_sma': closes[-1] > sma_20
                }
        except Exception as e:
            logger.debug(f"Macro error {ticker}: {e}")
    
    return results


def analyze_macro(index_data: Dict) -> Dict:
    """Analyze market regime"""
    if not index_data:
        return {'regime': 'NEUTRAL', 'trend_strength': 50, 'alignment': 50}
    
    bullish_count = 0
    bearish_count = 0
    returns = []
    
    for name, data in index_data.items():
        ret = data.get('ret_20d', 0)
        returns.append(ret)
        
        if ret > 2:
            bullish_count += 1
        elif ret < -2:
            bearish_count += 1
    
    if bullish_count >= 2:
        regime = 'BULLISH'
    elif bearish_count >= 2:
        regime = 'BEARISH'
    else:
        regime = 'NEUTRAL'
    
    avg_abs_ret = np.mean([abs(r) for r in returns]) if returns else 0
    trend_strength = min(avg_abs_ret * 10, 100)
    
    if len(returns) >= 2:
        all_positive = all(r > 0 for r in returns)
        all_negative = all(r < 0 for r in returns)
        alignment = 90 if (all_positive or all_negative) else 50
    else:
        alignment = 50
    
    return {
        'regime': regime,
        'trend_strength': trend_strength,
        'alignment': alignment,
        'details': index_data
    }

# =============================================================================
# REGIME ADJUSTER
# =============================================================================

def adjust_confidence(base_confidence: float, direction: str, macro: Dict) -> Tuple[float, str]:
    """Adjust confidence based on macro regime"""
    adjustments = []
    total_adj = 0
    
    regime = macro.get('regime', 'NEUTRAL')
    alignment = macro.get('alignment', 50) / 100
    
    if direction == 'LONG':
        if regime == 'BULLISH':
            adj = 15 * alignment
            total_adj += adj
            adjustments.append(f"Macro +{adj:.0f}%")
        elif regime == 'BEARISH':
            adj = -15 * alignment
            total_adj += adj
            adjustments.append(f"Macro {adj:.0f}%")
    elif direction == 'SHORT':
        if regime == 'BEARISH':
            adj = 15 * alignment
            total_adj += adj
            adjustments.append(f"Macro +{adj:.0f}%")
        elif regime == 'BULLISH':
            adj = -15 * alignment
            total_adj += adj
            adjustments.append(f"Macro {adj:.0f}%")
    
    adjusted = max(0, min(99, base_confidence + total_adj))
    reason = " | ".join(adjustments) if adjustments else "No adjustment"
    
    return adjusted, reason


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def load(path: str):
    """Load JSON file"""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save(path: str, data):
    """Save JSON file"""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_state() -> Dict:
    """Get today's state"""
    state = load(STATE_FILE)
    today = datetime.now().strftime('%Y-%m-%d')
    
    if not state or state.get('date') != today:
        state = {
            'date': today,
            'realized_pnl': 0.0,
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'macro': None,
            'done_for_day': False,
            'last_exit_time': None  # BUG FIX: Track last exit for cooldown
        }
        save(STATE_FILE, state)
    
    return state


def update_state(**kwargs):
    """Update state"""
    state = get_state()
    state.update(kwargs)
    save(STATE_FILE, state)


def get_positions() -> List[Dict]:
    """Get open positions"""
    return load(POS_FILE) or []


def save_positions(positions: List[Dict]):
    """Save positions"""
    save(POS_FILE, positions)


def log_trade(trade: Dict):
    """Log completed trade"""
    trades = load(TRADES_FILE) or []
    trades.append(trade)
    save(TRADES_FILE, trades)

# =============================================================================
# ROBINHOOD CLIENT
# =============================================================================

class LaurenTrader:
    """LAUREN autonomous trader"""
    
    def __init__(self):
        self.rh = None
        self.authenticated = False
        self.portfolio_value = 0
        self.buying_power = 0
        self.lauren_buying_power = 0  # LAUREN's 100% share (FULL YOLO)
        
    def login(self) -> bool:
        """Login to Robinhood"""
        try:
            import robin_stocks.robinhood as rh
            self.rh = rh
            
            result = rh.login(RH_USERNAME, RH_PASSWORD)
            if result:
                self.authenticated = True
                logger.info("✅ Logged into Robinhood")
                return True
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self):
        """Logout"""
        if self.rh:
            self.rh.logout()
        self.authenticated = False
    
    def update_buying_power(self):
        """Update buying power (LAUREN gets 50%)"""
        try:
            account = self.rh.profiles.load_account_profile()
            self.buying_power = float(account.get('buying_power', 0))
            self.lauren_buying_power = self.buying_power * BUYING_POWER_SHARE
            
            profile = self.rh.profiles.load_portfolio_profile()
            self.portfolio_value = float(profile.get('equity', 0))
        except Exception as e:
            logger.warning(f"Error getting buying power: {e}")
    
    def get_price(self, symbol: str) -> float:
        """Get current price"""
        try:
            q = self.rh.stocks.get_latest_price(symbol)
            if q and q[0]:
                return float(q[0])
        except:
            pass
        return 0
    
    def get_options(self) -> List[Dict]:
        """Fetch options chain"""
        opts = []
        today = datetime.now().date()
        
        for ticker in TICKERS:
            try:
                chains = self.rh.options.get_chains(ticker)
                if not chains:
                    continue
                
                for exp in chains.get('expiration_dates', []):
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    dte = (exp_date - today).days
                    
                    if dte < DTE_MIN or dte > DTE_MAX:
                        continue
                    
                    chain = self.rh.options.find_options_by_expiration(
                        ticker, expirationDate=exp, optionType='call'
                    )
                    
                    for o in (chain or []):
                        bid = float(o.get('bid_price', 0) or 0)
                        ask = float(o.get('ask_price', 0) or 0)
                        
                        opts.append({
                            'ticker': ticker,
                            'strike': float(o.get('strike_price', 0)),
                            'exp': exp,
                            'dte': dte,
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2,
                            'oi': int(o.get('open_interest', 0) or 0),
                            'opt_type': 'call'
                        })
                    
                    # ALSO fetch puts for SHORT signals
                    put_chain = self.rh.options.find_options_by_expiration(
                        ticker, expirationDate=exp, optionType='put'
                    )
                    
                    for o in (put_chain or []):
                        bid = float(o.get('bid_price', 0) or 0)
                        ask = float(o.get('ask_price', 0) or 0)
                        
                        opts.append({
                            'ticker': ticker,
                            'strike': float(o.get('strike_price', 0)),
                            'exp': exp,
                            'dte': dte,
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2,
                            'oi': int(o.get('open_interest', 0) or 0),
                            'opt_type': 'put'
                        })
            except Exception as e:
                logger.debug(f"Error fetching options for {ticker}: {e}")
        
        return opts
    
    def buy_option(self, ticker: str, strike: float, exp: str, 
                   quantity: int, limit_price: float, direction: str = 'LONG') -> Dict:
        """Buy option - CALL for LONG, PUT for SHORT"""
        
        # BUG FIX: SHORT = PUT, LONG = CALL
        option_type = 'put' if direction == 'SHORT' else 'call'
        symbol = 'P' if direction == 'SHORT' else 'C'
        
        logger.info(f"🎯 BUYING {quantity} {ticker} ${strike}{symbol} @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_buy_option_limit(
                positionEffect='open',
                creditOrDebit='debit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType=option_type,  # NOW DYNAMIC!
                timeInForce='gfd'
            )
            logger.info(f"   Order submitted: {result}")
            return result or {}
        except Exception as e:
            logger.error(f"   Buy error: {e}")
            return {'error': str(e)}
    
    def get_fill_price(self, order_id: str, max_wait: int = 30) -> Optional[float]:
        """Wait for order to fill and return actual fill price"""
        import time
        
        for _ in range(max_wait):
            try:
                order = self.rh.orders.get_option_order_info(order_id)
                if order:
                    state = order.get('state', '')
                    if state == 'filled':
                        # Get average fill price
                        avg_price = order.get('average_price')
                        if avg_price:
                            return float(avg_price)
                        # Fallback to price
                        price = order.get('price')
                        if price:
                            return float(price)
                    elif state in ['cancelled', 'rejected', 'failed']:
                        logger.warning(f"Order {state}")
                        return None
            except Exception as e:
                logger.debug(f"Error checking order: {e}")
            
            time.sleep(1)
        
        logger.warning(f"Order not filled after {max_wait}s")
        return None
    
    def sell_option(self, ticker: str, strike: float, exp: str,
                    quantity: int, limit_price: float, direction: str = 'LONG') -> Dict:
        """Sell option to close - CALL for LONG, PUT for SHORT"""
        
        option_type = 'put' if direction == 'SHORT' else 'call'
        symbol = 'P' if direction == 'SHORT' else 'C'
        
        logger.info(f"💰 SELLING {quantity} {ticker} ${strike}{symbol} @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_sell_option_limit(
                positionEffect='close',
                creditOrDebit='credit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType=option_type,  # NOW DYNAMIC!
                timeInForce='gfd'
            )
            logger.info(f"   Order submitted: {result}")
            return result or {}
        except Exception as e:
            logger.error(f"   Sell error: {e}")
            return {'error': str(e)}

# =============================================================================
# SCANNER
# =============================================================================

def scan_for_signals(opts: List[Dict], spots: Dict[str, float], macro: Dict, rh=None) -> List[Dict]:
    """Scan for trading signals - PATIENT, NO CHASING"""
    signals = []
    
    # Build a lookup for puts (to find matching put when SHORT signal detected)
    puts_lookup = {}
    for o in opts:
        if o.get('opt_type') == 'put':
            key = (o['ticker'], o['strike'], o['exp'])
            puts_lookup[key] = o
    
    for o in opts:
        # Only analyze CALLS for signal generation
        # (Quantum score is based on call pricing via Black-Scholes)
        if o.get('opt_type', 'call') != 'call':
            continue
        
        ticker = o['ticker']
        spot = spots.get(ticker)
        if not spot:
            continue
        
        strike = o['strike']
        mid = o['mid']
        bid = o['bid']
        dte = o['dte']
        oi = o['oi']
        ask = o['ask']
        volume = o.get('volume', 0)
        
        # =================================================================
        # QUALITY FILTERS - NO DUMB TRADES
        # =================================================================
        
        # Basic filters
        if mid < MIN_PREMIUM or bid < MIN_BID:
            continue
        
        # Liquidity filters
        if oi < MIN_OI:
            logger.debug(f"Skip {ticker} ${strike}: Low OI ({oi})")
            continue
        
        if volume < MIN_VOLUME:
            logger.debug(f"Skip {ticker} ${strike}: Low volume ({volume})")
            continue
        
        # Spread check - tight spreads only
        spread_pct = (ask - bid) / mid if mid > 0 else 999
        if spread_pct > MAX_SPREAD_PCT:
            logger.debug(f"Skip {ticker} ${strike}: Wide spread ({spread_pct:.0%})")
            continue
        
        # OTM range
        otm = (strike - spot) / spot
        if otm < MIN_OTM or otm > MAX_OTM:
            continue
        
        # =================================================================
        # SIGNAL CALCULATION
        # =================================================================
        
        T = dte / 365
        hv = get_historical_vol(rh, ticker)
        raw_score, iv = quantum_score(mid, spot, strike, T, hv)
        
        if raw_score is None:
            continue
        
        # =================================================================
        # ANTI-CHASE FILTER: IV SPIKE CHECK
        # =================================================================
        
        if iv and hv and iv > hv * MAX_IV_SPIKE:
            logger.debug(f"Skip {ticker} ${strike}: IV spike ({iv:.0%} > {hv*MAX_IV_SPIKE:.0%})")
            continue
        
        # =================================================================
        # DIRECTION + SCORE STRENGTH CHECK
        # =================================================================
        
        # Direction - BOTH LONG AND SHORT ENABLED
        if raw_score < LONG_THRESH:
            direction = 'LONG'
            score_strength = abs(raw_score - LONG_THRESH)
            
            # LONG = buy call, use call prices
            trade_bid = bid
            trade_ask = ask
            trade_mid = mid
            
        elif raw_score > SHORT_THRESH:
            direction = 'SHORT'
            score_strength = abs(raw_score - SHORT_THRESH)
            
            # SHORT = buy put, find matching put and use put prices
            put_key = (ticker, strike, o['exp'])
            matching_put = puts_lookup.get(put_key)
            
            if not matching_put:
                logger.debug(f"No matching put found for {ticker} ${strike} {o['exp']}")
                continue
            
            trade_bid = matching_put['bid']
            trade_ask = matching_put['ask']
            trade_mid = matching_put['mid']
            
            # Check put has good liquidity
            if trade_mid < MIN_PREMIUM or trade_bid < MIN_BID:
                continue
            
            put_spread = (trade_ask - trade_bid) / trade_mid if trade_mid > 0 else 999
            if put_spread > MAX_SPREAD_PCT:
                continue
        else:
            continue  # Score in neutral zone
        
        # =================================================================
        # PATIENCE FILTER: SCORE STRENGTH
        # =================================================================
        
        if score_strength < MIN_SCORE_STRENGTH:
            logger.debug(f"Skip {ticker} ${strike}: Weak signal (strength {score_strength:.2f} < {MIN_SCORE_STRENGTH})")
            continue
        
        # =================================================================
        # CONFIDENCE CHECK
        # =================================================================
        
        base_conf = score_to_confidence(raw_score)
        adj_conf, adj_reason = adjust_confidence(base_conf, direction, macro)
        
        if adj_conf < MIN_CONFIDENCE:
            continue
        
        # =================================================================
        # QUALITY SIGNAL - ADD TO LIST
        # =================================================================
        
        signals.append({
            'ticker': ticker,
            'strike': strike,
            'exp': o['exp'],
            'dte': dte,
            'bid': trade_bid,
            'ask': trade_ask,
            'mid': trade_mid,
            'spot': spot,
            'otm': otm * 100,
            'confidence': adj_conf,
            'adj_reason': adj_reason,
            'direction': direction,
            'score': raw_score,
            'score_strength': score_strength,
            'iv': iv,
            'hv': hv,
            'volume': volume,
            'oi': oi
        })
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


def check_exit(pos: Dict, opts: List[Dict]) -> Tuple[bool, str, float, float]:
    """
    Check if position should exit.
    
    EXIT HIERARCHY:
    1. STOP LOSS: -35% hard cut (no exceptions)
    2. TRAILING STOP: Locks in gains when activated
    3. PROFIT LOCK: At +100%, minimum +75% locked
    4. TIME STOP: Exit at 2 DTE
    """
    curr = None
    pos_direction = pos.get('direction', 'LONG')
    expected_opt_type = 'put' if pos_direction == 'SHORT' else 'call'
    
    # Find current price
    for o in opts:
        if (o['ticker'] == pos['ticker'] and 
            o['strike'] == pos['strike'] and 
            o['exp'] == pos['exp'] and
            o.get('opt_type', 'call') == expected_opt_type):
            curr = o['mid']
            break
    
    if curr is None:
        exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
        if datetime.now().date() >= exp:
            return True, 'EXPIRY', -100, 0
        return False, '', 0, 0
    
    entry = pos['entry']
    pnl = (curr - entry) / entry * 100
    
    # Track high water mark for trailing stop
    high_water = pos.get('high_water_pnl', pnl)
    if pnl > high_water:
        high_water = pnl
    
    # =========================================================================
    # EXIT CHECK 1: STOP LOSS (-35%) - HARD CUT
    # =========================================================================
    if pnl <= STOP:
        logger.info(f"🛑 STOP LOSS HIT: {pnl:.0f}% <= {STOP}%")
        return True, 'STOP', pnl, curr
    
    # =========================================================================
    # EXIT CHECK 2: PROFIT LOCK (at +100%, lock in +75%)
    # =========================================================================
    if high_water >= PROFIT_LOCK_THRESH:
        lock_level = PROFIT_LOCK_MIN
        if pnl <= lock_level:
            logger.info(f"🔒 PROFIT LOCK: Was +{high_water:.0f}%, locking in +{lock_level}%")
            return True, 'PROFIT_LOCK', pnl, curr
    
    # =========================================================================
    # EXIT CHECK 3: TRAILING STOP
    # =========================================================================
    if TRAILING_STOP and high_water >= TRAIL_ACTIVATE:
        # Trail stop is high_water - TRAIL_DISTANCE
        trail_stop = high_water - TRAIL_DISTANCE
        
        if pnl <= trail_stop:
            logger.info(f"📈 TRAILING STOP: Peak +{high_water:.0f}%, trail at +{trail_stop:.0f}%, now +{pnl:.0f}%")
            return True, 'TRAIL_STOP', pnl, curr
    
    # =========================================================================
    # EXIT CHECK 4: TIME STOP (theta protection)
    # =========================================================================
    exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
    if (exp - datetime.now().date()).days <= TIME_STOP_DTE:
        logger.info(f"⏰ TIME STOP: {TIME_STOP_DTE} DTE, exiting with {pnl:+.0f}%")
        return True, 'TIME_STOP', pnl, curr
    
    # Update high water mark in position (caller should save)
    pos['high_water_pnl'] = high_water
    
    return False, '', pnl, curr

# =============================================================================
# MAIN LOOP
# =============================================================================

def run():
    """Main trading loop"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██╗      █████╗ ██╗   ██╗██████╗ ███████╗███╗   ██╗                      ║
║     ██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝████╗  ██║                      ║
║     ██║     ███████║██║   ██║██████╔╝█████╗  ██╔██╗ ██║                      ║
║     ██║     ██╔══██║██║   ██║██╔══██╗██╔══╝  ██║╚██╗██║                      ║
║     ███████╗██║  ██║╚██████╔╝██║  ██║███████╗██║ ╚████║                      ║
║     ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝                      ║
║                                                                              ║
║     🚀 ACCOUNT FLIP MODE 🚀                                                  ║
║                                                                              ║
║     Strategy: Patient entries, let winners ride, cut losers fast             ║
║     Stop Loss: -35% (STRICT)                                                 ║
║     Trailing Stop: Activates at +40%, trails by 25%                          ║
║     Profit Lock: At +100%, locks in +75% minimum                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    trader = LaurenTrader()
    
    if not trader.login():
        logger.error("Failed to login!")
        return
    
    trader.update_buying_power()
    logger.info(f"💰 Total Buying Power: ${trader.buying_power:.2f}")
    logger.info(f"🚀 LAUREN's Share (100%): ${trader.lauren_buying_power:.2f}")
    logger.info(f"📈 Tickers: {', '.join(TICKERS)}")
    logger.info(f"⚡ Mode: AGGRO (LONG + SHORT)")
    
    
    while True:
        try:
            now = datetime.now()
            hour = now.hour
            weekday = now.weekday()
            
            # Market hours
            if weekday >= 5:
                logger.debug("Weekend - sleeping...")
                time.sleep(3600)
                continue
            
            if hour < 9 or (hour == 9 and now.minute < 35):
                logger.debug("Market not open yet...")
                time.sleep(60)
                continue
            
            if hour >= 16:
                logger.debug("Market closed...")
                time.sleep(3600)
                continue
            
            # Check daily limits
            state = get_state()
            if state.get('done_for_day'):
                logger.info("Done for today!")
                time.sleep(3600)
                continue
            
            if state['realized_pnl'] >= DAILY_PROFIT_TARGET:
                logger.info(f"🎯 Daily target reached: +${state['realized_pnl']:.2f}")
                update_state(done_for_day=True)
                continue
            
            if state['realized_pnl'] <= -DAILY_LOSS_LIMIT:
                logger.info(f"🛑 Daily stop hit: -${abs(state['realized_pnl']):.2f}")
                update_state(done_for_day=True)
                continue
            
            # Update buying power
            trader.update_buying_power()
            
            # Fetch data
            spots = {}
            for t in TICKERS:
                p = trader.get_price(t)
                if p > 0:
                    spots[t] = p
            
            if not spots:
                time.sleep(60)
                continue
            
            opts = trader.get_options()
            positions = get_positions()
            
            # Fetch macro (once per day)
            macro = state.get('macro')
            if not macro:
                logger.info("Fetching macro data...")
                index_data = fetch_index_data(trader.rh)
                macro = analyze_macro(index_data)
                update_state(macro=macro)
                logger.info(f"Market Regime: {macro['regime']}")
            
            # Log status
            logger.info("")
            logger.info(f"{'='*50}")
            logger.info(f"CYCLE: {now.strftime('%H:%M:%S')}")
            logger.info(f"{'='*50}")
            logger.info(f"NVDL: ${spots.get('NVDL', 0):.2f} | TSLL: ${spots.get('TSLL', 0):.2f}")
            logger.info(f"Regime: {macro.get('regime', 'NEUTRAL')}")
            logger.info(f"Positions: {len(positions)} | P&L: ${state['realized_pnl']:+.2f}")
            
            # === CHECK POSITIONS FOR EXITS ===
            new_positions = []
            for pos in positions:
                exit_flag, reason, pnl, curr = check_exit(pos, opts)
                
                if exit_flag:
                    logger.info(f"")
                    pos_symbol = 'P' if pos.get('direction') == 'SHORT' else 'C'
                    emoji = "💰" if pnl > 0 else "📉"
                    logger.info(f"{emoji} CLOSING: {pos['ticker']} ${pos['strike']}{pos_symbol} | {reason} | {pnl:+.0f}%")
                    
                    # Execute sell
                    trader.sell_option(
                        pos['ticker'],
                        pos['strike'],
                        pos['exp'],
                        pos['contracts'],
                        curr * 0.98,  # Slightly below to fill
                        pos.get('direction', 'LONG')  # Pass direction for PUT/CALL
                    )
                    
                    # Calculate actual P&L
                    actual_pnl = (curr - pos['entry']) * pos['contracts'] * 100
                    
                    # Update state + COOLDOWN
                    state = get_state()
                    new_realized = state['realized_pnl'] + actual_pnl
                    new_trades = state['trades_today'] + 1
                    new_wins = state['wins'] + (1 if pnl > 0 else 0)
                    new_losses = state['losses'] + (1 if pnl <= 0 else 0)
                    update_state(
                        realized_pnl=new_realized,
                        trades_today=new_trades,
                        wins=new_wins,
                        losses=new_losses,
                        last_exit_time=datetime.now().isoformat()  # BUG FIX: Track exit time
                    )
                    
                    # Log trade
                    log_trade({
                        'date': now.strftime('%Y-%m-%d %H:%M'),
                        'ticker': pos['ticker'],
                        'strike': pos['strike'],
                        'direction': pos.get('direction', 'LONG'),
                        'entry': pos['entry'],
                        'exit': curr,
                        'contracts': pos['contracts'],
                        'pnl_pct': pnl,
                        'pnl_dollar': actual_pnl,
                        'reason': reason
                    })
                else:
                    new_positions.append(pos)
                    if curr:
                        pos_symbol = 'P' if pos.get('direction') == 'SHORT' else 'C'
                        logger.info(f"   Position: {pos['ticker']} ${pos['strike']}{pos_symbol} | {pnl:+.0f}%")
            
            save_positions(new_positions)
            positions = new_positions
            
            # === SCAN FOR NEW ENTRIES ===
            # BUG FIX: Check cooldown after exit (no flip-flop)
            COOLDOWN_SECONDS = 300  # 5 minutes
            state = get_state()
            last_exit = state.get('last_exit_time')
            if last_exit:
                try:
                    exit_time = datetime.fromisoformat(last_exit)
                    seconds_since_exit = (datetime.now() - exit_time).total_seconds()
                    if seconds_since_exit < COOLDOWN_SECONDS:
                        remaining = int(COOLDOWN_SECONDS - seconds_since_exit)
                        logger.info(f"   ⏳ Cooldown: {remaining}s until next entry allowed")
                        continue  # Skip entry scan
                except:
                    pass
            
            # =================================================================
            # PATIENCE CHECKS - DON'T TAKE DUMB/LATE TRADES
            # =================================================================
            
            # Check daily trade limit
            if state['trades_today'] >= MAX_TRADES_PER_DAY:
                logger.info(f"   📊 Daily trade limit reached ({MAX_TRADES_PER_DAY})")
                continue
            
            # Check time of day - avoid first 15 min (wild spreads)
            market_open = now.replace(hour=6, minute=30, second=0, microsecond=0)  # 9:30 ET = 6:30 PT
            minutes_since_open = (now - market_open).total_seconds() / 60
            
            if AVOID_FIRST_15_MIN and minutes_since_open < 15:
                logger.info(f"   ⏰ Waiting for market to settle ({15 - int(minutes_since_open)} min)")
                continue
            
            # Check time of day - avoid last 30 min (can't manage)
            market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)  # 4:00 ET = 1:00 PT
            minutes_to_close = (market_close - now).total_seconds() / 60
            
            if minutes_to_close < MIN_TIME_TO_CLOSE:
                logger.info(f"   ⏰ Too close to market close ({int(minutes_to_close)} min)")
                continue
            
            # =================================================================
            # SCAN FOR QUALITY SIGNALS
            # =================================================================
            
            if len(positions) < MAX_POSITIONS:
                signals = scan_for_signals(opts, spots, macro, trader.rh)
                
                if signals:
                    best = signals[0]
                    
                    # Check we don't already have this
                    already_have = any(
                        p['ticker'] == best['ticker'] and 
                        p['strike'] == best['strike']
                        for p in positions
                    )
                    
                    if not already_have:
                        # Calculate position size
                        max_spend = trader.lauren_buying_power * MAX_POSITION_PCT
                        contract_cost = best['mid'] * 100  # Use mid for sizing
                        contracts = max(1, int(max_spend / contract_cost))
                        
                        # Cap at available buying power
                        total_cost = contracts * contract_cost
                        if total_cost > trader.lauren_buying_power * 0.95:
                            contracts = max(1, int(trader.lauren_buying_power * 0.95 / contract_cost))
                        
                        if contracts >= 1 and contract_cost <= trader.lauren_buying_power:
                            option_symbol = 'P' if best['direction'] == 'SHORT' else 'C'
                            logger.info(f"")
                            logger.info(f"🎯 SIGNAL: {best['ticker']} ${best['strike']}{option_symbol} ({best['direction']})")
                            logger.info(f"   Score: {best['score']:.2f} | Strength: {best.get('score_strength', 0):.2f}")
                            logger.info(f"   IV: {best.get('iv', 0)*100:.0f}% | HV: {best.get('hv', 0)*100:.0f}%")
                            logger.info(f"   Confidence: {best['confidence']:.0f}% | {best['adj_reason']}")
                            logger.info(f"   Volume: {best.get('volume', 0):,} | OI: {best.get('oi', 0):,}")
                            logger.info(f"   Bid: ${best['bid']:.2f} | Ask: ${best['ask']:.2f} | Mid: ${best['mid']:.2f}")
                            
                            # Get volume/OI for liquidity calculation
                            volume = best.get('volume', 0) or 0
                            oi = best.get('oi', 0) or 0
                            
                            # ALGO'S DESIRED FILL PRICE = MID
                            desired_fill = best['mid']
                            max_price = best['ask']
                            
                            # Try up to 3 times with smart price adjustment
                            max_attempts = 3
                            fill_successful = False
                            actual_entry = None
                            
                            for attempt in range(max_attempts):
                                if attempt == 0:
                                    # First try: exact desired price (mid)
                                    try_price = desired_fill
                                    price_reasoning = "Mid price"
                                else:
                                    # Re-evaluate and calculate optimal entry
                                    logger.info(f"")
                                    logger.info(f"   🔄 ATTEMPT {attempt + 1}: Recalculating optimal entry...")
                                    
                                    # Recalculate confidence with current market data
                                    T = best['dte'] / 365
                                    hv = get_historical_vol(trader.rh, best['ticker'])  # DYNAMIC VOL
                                    new_score, _ = quantum_score(best['mid'], best['spot'], best['strike'], T, hv)
                                    
                                    if new_score is None:
                                        logger.info(f"   ❌ Cannot recalculate - aborting")
                                        break
                                    
                                    new_confidence = score_to_confidence(new_score)
                                    
                                    if new_confidence < 65:
                                        logger.info(f"   ❌ Confidence dropped to {new_confidence:.0f}% - aborting")
                                        break
                                    
                                    # Calculate optimal entry using all factors
                                    try_price, price_reasoning = calculate_optimal_entry(
                                        desired_price=desired_fill,
                                        ask_price=best['ask'],
                                        bid_price=best['bid'],
                                        confidence=new_confidence,
                                        spot_price=best['spot'],
                                        strike=best['strike'],
                                        dte=best['dte'],
                                        volume=volume,
                                        open_interest=oi
                                    )
                                    
                                    logger.info(f"   Confidence: {new_confidence:.0f}%")
                                    logger.info(f"   Factors: {price_reasoning}")
                                    logger.info(f"   Optimal entry: ${desired_fill:.2f} → ${try_price:.2f}")
                                    
                                    if try_price <= desired_fill:
                                        logger.info(f"   Model says don't pay more - aborting")
                                        break
                                    
                                    if try_price >= max_price:
                                        logger.info(f"   ⚠️ At max price (ask), final attempt")
                                
                                logger.info(f"")
                                logger.info(f"   📌 DESIRED FILL: ${desired_fill:.2f}")
                                logger.info(f"   📌 TRYING PRICE: ${try_price:.2f} ({price_reasoning})")
                                logger.info(f"   Sending limit order for {contracts} contracts")
                                
                                # Execute buy
                                order_result = trader.buy_option(
                                    best['ticker'],
                                    best['strike'],
                                    best['exp'],
                                    contracts,
                                    try_price,
                                    best['direction']  # BUG FIX: Pass direction for PUT/CALL
                                )
                                
                                order_id = order_result.get('id')
                                if not order_id:
                                    logger.warning(f"   No order ID returned")
                                    break
                                
                                logger.info(f"   Waiting for fill...")
                                fill_price = trader.get_fill_price(order_id, max_wait=15)
                                
                                if fill_price:
                                    # Check fill is at or below our try price
                                    if fill_price > try_price * 1.01:
                                        logger.error(f"   ❌ REJECTED - filled above limit")
                                        logger.error(f"   Tried: ${try_price:.2f}, Got: ${fill_price:.2f}")
                                        try:
                                            trader.rh.orders.cancel_option_order(order_id)
                                        except:
                                            pass
                                        break
                                    
                                    actual_entry = fill_price
                                    fill_successful = True
                                    
                                    logger.info(f"")
                                    logger.info(f"   ✅ FILLED @ ${fill_price:.2f}")
                                    if fill_price > desired_fill:
                                        premium_paid = (fill_price - desired_fill) / desired_fill * 100
                                        logger.info(f"   Premium paid: +{premium_paid:.1f}% over mid")
                                    elif fill_price < desired_fill:
                                        savings = (desired_fill - fill_price) / desired_fill * 100
                                        logger.info(f"   Savings: -{savings:.1f}% below mid")
                                    break
                                else:
                                    logger.info(f"   ❌ No fill at ${try_price:.2f}")
                                    try:
                                        trader.rh.orders.cancel_option_order(order_id)
                                    except:
                                        pass
                                    
                                    if attempt < max_attempts - 1:
                                        logger.info(f"   Recalculating optimal entry...")
                            
                            if not fill_successful:
                                logger.info(f"   ❌ Could not fill - skipping")
                                continue
                            
                            # Calculate TP/SL based on ACTUAL entry
                            tp_price = actual_entry * 1.50  # +50%
                            sl_price = actual_entry * 0.60  # -40%
                            logger.info(f"")
                            logger.info(f"   📊 POSITION OPENED:")
                            logger.info(f"   Entry: ${actual_entry:.2f}")
                            logger.info(f"   TP:    ${tp_price:.2f} (+50%)")
                            logger.info(f"   SL:    ${sl_price:.2f} (-40%)")
                            
                            positions.append({
                                'ticker': best['ticker'],
                                'strike': best['strike'],
                                'exp': best['exp'],
                                'entry': actual_entry,
                                'contracts': contracts,
                                'confidence': best['confidence'],
                                'entry_time': now.isoformat(),
                                'direction': best['direction']  # BUG FIX: Track direction
                            })
                            save_positions(positions)
            
            # Sleep until next scan
            logger.info(f"")
            logger.info(f"Next scan in {SCAN_INTERVAL // 60} min...")
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)
    
    trader.logout()


if __name__ == "__main__":
    import sys
    
    if '--trades' in sys.argv:
        trades = load(TRADES_FILE) or []
        for t in trades:
            w = "W" if t['pnl_pct'] > 0 else "L"
            print(f"{t['date']} | {t['ticker']} ${t['strike']}C | {t['pnl_pct']:+.0f}% | ${t['pnl_dollar']:+.2f} {w}")
        if trades:
            wins = len([t for t in trades if t['pnl_pct'] > 0])
            total_pnl = sum(t['pnl_dollar'] for t in trades)
            print(f"\n{wins}/{len(trades)} wins ({wins/len(trades)*100:.0f}%) | Total: ${total_pnl:+.2f}")
    elif '--clear' in sys.argv:
        for f in [POS_FILE, STATE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleared")
    else:
        run()
