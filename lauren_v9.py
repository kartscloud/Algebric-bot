#!/usr/bin/env python3
"""
================================================================================
LAUREN v9 - One Trade Per Day + Proper Shorting
================================================================================

Changes from v8 AGGRO:
- 100% position sizing (all in)
- 35% max loss based on position cost (not fixed dollar)
- One trade per day - find the best signal, trade it, done
- FIXED: SHORT signals now buy PUTS (was incorrectly buying calls)

Strategy:
- LONG signal (underpriced) → Buy CALL
- SHORT signal (overpriced) → Buy PUT

================================================================================
"""

import os
import json
import time
from datetime import datetime, timedelta
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

RH_USERNAME = "your_email@example.com"
RH_PASSWORD = "your_password"

TICKERS = ['NVDL', 'TSLL']

UNDERLYING = {
    'NVDL': 'NVDA',
    'TSLL': 'TSLA'
}

LEVERAGE = {
    'NVDL': 2.0,
    'TSLL': 2.0
}

# =============================================================================
# V9 POSITION SIZING - ALL IN, ONE TRADE
# =============================================================================

MAX_POSITIONS = 1                  # Single position
MAX_POSITION_PCT = 1.00            # 100% of capital per trade
MAX_LOSS_PCT = 0.35                # 35% max loss (of position cost)
MAX_DAILY_PROFIT = 2000.0          # Lock in gains
MIN_BUYING_POWER_RESERVE = 25.0    # Minimal reserve

# =============================================================================
# SIGNAL THRESHOLDS
# =============================================================================

SCAN_INTERVAL = 180
LONG_THRESH = -0.30
SHORT_THRESH = 0.40
MIN_CONFIDENCE = 50

# =============================================================================
# OPTION FILTERS
# =============================================================================

DTE_MIN = 5
DTE_MAX = 14
MIN_OTM = 0.05
MAX_OTM = 0.20
MIN_PREMIUM = 0.10
MIN_BID = 0.05
MIN_OI = 25
MAX_SPREAD_PCT = 0.15

# =============================================================================
# EXIT RULES
# =============================================================================

TARGET_PCT = 50                    # Take profit at +50%
STOP_PCT = -35                     # Stop loss at -35% (matches max loss)
TIME_STOP_DTE = 1

# =============================================================================
# CIRCUIT BREAKERS
# =============================================================================

MARKET_OPEN_DELAY_MINUTES = 5
MAX_CONSECUTIVE_REJECTS = 5
MAX_SPREAD_BLOWOUT_PCT = 0.30
MIN_QUOTE_FRESHNESS_SEC = 120

# =============================================================================
# VOLATILITY
# =============================================================================

VOL_LOOKBACK_DAYS = 20
VOL_MIN = 0.30
VOL_MAX = 2.00

# =============================================================================
# FILES
# =============================================================================

DATA_DIR = "lauren_v9_data"
LOG_DIR = "lauren_v9_logs"
POS_FILE = f"{DATA_DIR}/positions.json"
STATE_FILE = f"{DATA_DIR}/state.json"
TRADES_FILE = f"{DATA_DIR}/trades.json"
VOL_CACHE_FILE = f"{DATA_DIR}/vol_cache.json"

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | LAUREN-v9 | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{LOG_DIR}/lauren.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# BLACK-SCHOLES (supports both calls and puts)
# =============================================================================

def bs_call(S: float, K: float, T: float, vol: float, r: float = 0.05) -> float:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return max(0, S - K)
    try:
        d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        return S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    except:
        return max(0, S - K)


def bs_put(S: float, K: float, T: float, vol: float, r: float = 0.05) -> float:
    if T <= 0 or vol <= 0 or S <= 0 or K <= 0:
        return max(0, K - S)
    try:
        d1 = (log(S/K) + (r + 0.5*vol**2)*T) / (vol*sqrt(T))
        d2 = d1 - vol*sqrt(T)
        return K * exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    except:
        return max(0, K - S)


def calc_iv_call(price: float, S: float, K: float, T: float) -> Optional[float]:
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    try:
        return brentq(lambda v: bs_call(S, K, T, v) - price, 0.01, 5.0, maxiter=100)
    except:
        return None


def calc_iv_put(price: float, S: float, K: float, T: float) -> Optional[float]:
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    try:
        return brentq(lambda v: bs_put(S, K, T, v) - price, 0.01, 5.0, maxiter=100)
    except:
        return None


def quantum_score_call(price: float, S: float, K: float, T: float, hv: float) -> Tuple[Optional[float], Optional[float]]:
    """Score for CALL options - negative = underpriced = LONG"""
    iv = calc_iv_call(price, S, K, T)
    if not iv or iv < 0.1 or iv > 3:
        return None, None
    bs = bs_call(S, K, T, hv)
    if bs < 0.01:
        return None, None
    score = 0.4 * (iv - hv) / hv + 0.6 * (price - bs) / bs
    return score, iv


def quantum_score_put(price: float, S: float, K: float, T: float, hv: float) -> Tuple[Optional[float], Optional[float]]:
    """Score for PUT options - positive = overpriced SHORT signal, negative = underpriced"""
    iv = calc_iv_put(price, S, K, T)
    if not iv or iv < 0.1 or iv > 3:
        return None, None
    bs = bs_put(S, K, T, hv)
    if bs < 0.01:
        return None, None
    score = 0.4 * (iv - hv) / hv + 0.6 * (price - bs) / bs
    return score, iv


def score_to_confidence(score: float) -> float:
    abs_score = abs(score)
    if abs_score <= 0.30:
        return 40 + (abs_score / 0.30) * 10
    elif abs_score <= 0.60:
        return 50 + ((abs_score - 0.30) / 0.30) * 25
    elif abs_score <= 1.0:
        return 75 + ((abs_score - 0.60) / 0.40) * 20
    return 95


# =============================================================================
# DYNAMIC VOLATILITY
# =============================================================================

def calculate_realized_vol(rh, ticker: str) -> float:
    underlying = UNDERLYING.get(ticker, ticker)
    leverage = LEVERAGE.get(ticker, 1.0)
    
    try:
        historicals = rh.stocks.get_stock_historicals(
            underlying, interval='day', span='month', bounds='regular'
        )
        
        if not historicals or len(historicals) < 10:
            return 0.65 * leverage
        
        closes = [float(h['close_price']) for h in historicals if h.get('close_price')]
        if len(closes) < 10:
            return 0.65 * leverage
        
        returns = np.diff(np.log(closes))
        daily_vol = np.std(returns)
        annual_vol = daily_vol * sqrt(252)
        leveraged_vol = annual_vol * leverage
        leveraged_vol = max(VOL_MIN, min(VOL_MAX, leveraged_vol))
        
        return leveraged_vol
        
    except Exception as e:
        logger.warning(f"Vol calc error: {e}")
        return 0.65 * leverage


def get_dynamic_vol(rh, ticker: str) -> float:
    cache = load(VOL_CACHE_FILE) or {}
    cache_key = f"{ticker}_vol"
    cache_time_key = f"{ticker}_vol_time"
    
    if cache_key in cache and cache_time_key in cache:
        cache_time = datetime.fromisoformat(cache[cache_time_key])
        if datetime.now() - cache_time < timedelta(hours=1):
            return cache[cache_key]
    
    vol = calculate_realized_vol(rh, ticker)
    cache[cache_key] = vol
    cache[cache_time_key] = datetime.now().isoformat()
    save(VOL_CACHE_FILE, cache)
    
    return vol


# =============================================================================
# CIRCUIT BREAKERS
# =============================================================================

class CircuitBreaker:
    def __init__(self):
        self.consecutive_rejects = 0
        self.last_prices = {}
        self.is_tripped = False
        self.trip_reason = ""
    
    def check_market_open_delay(self) -> Tuple[bool, str]:
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now < market_open:
            return False, "Market not open"
        
        minutes_since_open = (now - market_open).total_seconds() / 60
        if minutes_since_open < MARKET_OPEN_DELAY_MINUTES:
            return False, f"Chaos window: {MARKET_OPEN_DELAY_MINUTES - minutes_since_open:.0f}m left"
        
        return True, ""
    
    def check_spread(self, bid: float, ask: float, mid: float) -> Tuple[bool, str]:
        if mid <= 0:
            return False, "Invalid mid"
        spread_pct = (ask - bid) / mid
        if spread_pct > MAX_SPREAD_BLOWOUT_PCT:
            return False, f"Spread blowout: {spread_pct*100:.0f}%"
        return True, ""
    
    def check_quote_freshness(self, ticker: str, price: float) -> Tuple[bool, str]:
        now = datetime.now()
        if ticker in self.last_prices:
            last_price, last_time = self.last_prices[ticker]
            time_diff = (now - last_time).total_seconds()
            if time_diff > MIN_QUOTE_FRESHNESS_SEC:
                price_change = abs(price - last_price) / last_price if last_price > 0 else 0
                if price_change < 0.001:
                    return False, f"Stale quote: {time_diff:.0f}s"
        self.last_prices[ticker] = (price, now)
        return True, ""
    
    def record_reject(self):
        self.consecutive_rejects += 1
        if self.consecutive_rejects >= MAX_CONSECUTIVE_REJECTS:
            self.is_tripped = True
            self.trip_reason = f"Too many rejects: {self.consecutive_rejects}"
    
    def record_fill(self):
        self.consecutive_rejects = 0
    
    def reset(self):
        self.consecutive_rejects = 0
        self.last_prices = {}
        self.is_tripped = False
        self.trip_reason = ""
    
    def is_safe_to_trade(self) -> Tuple[bool, str]:
        if self.is_tripped:
            return False, f"Tripped: {self.trip_reason}"
        return self.check_market_open_delay()


# =============================================================================
# REALISTIC FILL MODELS
# =============================================================================

def calculate_entry_fill(
    mid: float,
    bid: float,
    ask: float,
    confidence: float,
    dte: int,
    open_interest: int,
    spread_pct: float
) -> Tuple[float, str]:
    """Realistic entry fill with slippage."""
    if confidence >= 90:
        conf_slip = 0.50
    elif confidence >= 70:
        conf_slip = 0.30 * ((confidence - 70) / 20)
    else:
        conf_slip = 0
    
    if open_interest < 100:
        oi_slip = 0.25
    elif open_interest < 300:
        oi_slip = 0.15
    else:
        oi_slip = 0.05
    
    if dte <= 5:
        time_slip = 0.20
    elif dte <= 7:
        time_slip = 0.10
    else:
        time_slip = 0
    
    if spread_pct > 0.12:
        spread_slip = 0.15
    elif spread_pct > 0.08:
        spread_slip = 0.08
    else:
        spread_slip = 0
    
    total_slip = min(conf_slip + oi_slip + time_slip + spread_slip, 0.80)
    spread = ask - bid
    
    fill = mid + (spread * total_slip * 0.5)
    fill = min(fill, ask)
    
    reasoning = f"Conf:{conf_slip:.0%} OI:{oi_slip:.0%} Time:{time_slip:.0%} Sprd:{spread_slip:.0%}"
    
    return round(fill, 2), reasoning


def calculate_exit_fill(bid: float, ask: float, urgency: str) -> float:
    """Realistic exit fill anchored to bid."""
    mid = (bid + ask) / 2
    
    if urgency == 'stop':
        return round(bid + (mid - bid) * 0.15, 2)
    elif urgency == 'time':
        return round(bid + (mid - bid) * 0.35, 2)
    else:  # target
        return round(bid + (mid - bid) * 0.55, 2)


# =============================================================================
# FILE HELPERS
# =============================================================================

def load(filepath: str) -> Optional[Dict]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def save(filepath: str, data: Dict):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def get_state() -> Dict:
    state = load(STATE_FILE)
    today = datetime.now().strftime('%Y-%m-%d')
    
    if not state or state.get('date') != today:
        state = {
            'date': today,
            'realized_pnl': 0.0,
            'trades_today': 0,
            'wins': 0,
            'losses': 0,
            'done_for_day': False,
            'traded_today': False  # NEW: Track if we've traded today
        }
        save(STATE_FILE, state)
    
    return state


def update_state(**kwargs):
    state = get_state()
    state.update(kwargs)
    save(STATE_FILE, state)


def get_positions() -> List[Dict]:
    return load(POS_FILE) or []


def save_positions(positions: List[Dict]):
    save(POS_FILE, positions)


def log_trade(trade: Dict):
    trades = load(TRADES_FILE) or []
    trades.append(trade)
    save(TRADES_FILE, trades)


# =============================================================================
# POSITION OBJECT
# =============================================================================

def create_position(
    ticker: str,
    strike: float,
    exp: str,
    entry_price: float,
    contracts: int,
    confidence: float,
    direction: str,
    option_type: str  # 'call' or 'put'
) -> Dict:
    tp_price = round(entry_price * (1 + TARGET_PCT / 100), 2)
    sl_price = round(entry_price * (1 + STOP_PCT / 100), 2)
    
    return {
        'ticker': ticker,
        'strike': strike,
        'exp': exp,
        'direction': direction,
        'option_type': option_type,  # NEW: Track if it's a call or put
        'entry_price': entry_price,
        'contracts': contracts,
        'confidence': confidence,
        'entry_time': datetime.now().isoformat(),
        'tp_price': tp_price,
        'sl_price': sl_price,
        'tp_pct': TARGET_PCT,
        'sl_pct': STOP_PCT,
        'cost': contracts * entry_price * 100
    }


# =============================================================================
# V9 POSITION SIZING - 100% WITH 35% MAX LOSS
# =============================================================================

def calculate_position_size(
    buying_power: float,
    entry_price: float
) -> int:
    """
    V9 sizing: 100% of capital, with 35% max loss based on position cost.
    """
    if entry_price <= 0:
        return 0
    
    cost_per_contract = entry_price * 100
    
    # Available capital (100% minus small reserve)
    available = buying_power - MIN_BUYING_POWER_RESERVE
    if available <= 0:
        return 0
    
    # Max contracts we can afford
    max_contracts = int(available / cost_per_contract)
    
    # The 35% max loss is built into the stop loss, not position sizing
    # So we just go all-in with what we can afford
    
    return max(1, max_contracts)


# =============================================================================
# ROBINHOOD CLIENT
# =============================================================================

class LaurenTrader:
    def __init__(self):
        self.rh = None
        self.authenticated = False
        self.buying_power = 0
        self.circuit_breaker = CircuitBreaker()
        
    def login(self) -> bool:
        try:
            import robin_stocks.robinhood as rh
            self.rh = rh
            result = rh.login(RH_USERNAME, RH_PASSWORD)
            if result:
                self.authenticated = True
                logger.info("✅ Logged in")
                return True
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def logout(self):
        if self.rh:
            self.rh.logout()
        self.authenticated = False
    
    def update_buying_power(self):
        try:
            account = self.rh.profiles.load_account_profile()
            self.buying_power = float(account.get('buying_power', 0))
        except Exception as e:
            logger.warning(f"BP error: {e}")
    
    def get_price(self, symbol: str) -> float:
        try:
            q = self.rh.stocks.get_latest_price(symbol)
            if q and q[0]:
                return float(q[0])
        except:
            pass
        return 0
    
    def get_options(self, option_type: str = 'call') -> List[Dict]:
        """Fetch options - now supports both calls and puts."""
        opts = []
        today = datetime.now().date()
        
        for ticker in TICKERS:
            try:
                spot = self.get_price(ticker)
                if spot <= 0:
                    continue
                
                chains = self.rh.options.get_chains(ticker)
                if not chains:
                    continue
                
                exp_dates = chains.get('expiration_dates', [])
                
                for exp in exp_dates:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                    dte = (exp_date - today).days
                    
                    if dte < DTE_MIN or dte > DTE_MAX:
                        continue
                    
                    options = self.rh.options.find_options_by_expiration(
                        [ticker], exp, optionType=option_type
                    )
                    
                    for o in options:
                        try:
                            strike = float(o.get('strike_price', 0))
                            bid = float(o.get('bid_price', 0) or 0)
                            ask = float(o.get('ask_price', 0) or 0)
                            oi = int(o.get('open_interest', 0) or 0)
                            volume = int(o.get('volume', 0) or 0)
                            
                            if bid <= 0 or ask <= 0:
                                continue
                            
                            mid = (bid + ask) / 2
                            
                            opts.append({
                                'ticker': ticker,
                                'strike': strike,
                                'exp': exp,
                                'dte': dte,
                                'bid': bid,
                                'ask': ask,
                                'mid': mid,
                                'oi': oi,
                                'volume': volume,
                                'spot': spot,
                                'option_type': option_type
                            })
                        except:
                            continue
                            
            except Exception as e:
                logger.debug(f"Options error {ticker}: {e}")
        
        return opts
    
    def get_all_options(self) -> List[Dict]:
        """Fetch both calls and puts."""
        calls = self.get_options('call')
        puts = self.get_options('put')
        return calls + puts
    
    def buy_option(self, ticker: str, strike: float, exp: str, 
                   quantity: int, limit_price: float, option_type: str) -> Dict:
        """Buy option - now properly handles calls vs puts."""
        type_label = 'C' if option_type == 'call' else 'P'
        logger.info(f"🎯 BUY {quantity} {ticker} ${strike}{type_label} @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_buy_option_limit(
                positionEffect='open',
                creditOrDebit='debit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType=option_type,  # NOW CORRECTLY PASSES call OR put
                timeInForce='gfd'
            )
            return result or {}
        except Exception as e:
            logger.error(f"Buy error: {e}")
            self.circuit_breaker.record_reject()
            return {'error': str(e)}
    
    def sell_option(self, ticker: str, strike: float, exp: str,
                    quantity: int, limit_price: float, option_type: str) -> Dict:
        """Sell option - now properly handles calls vs puts."""
        type_label = 'C' if option_type == 'call' else 'P'
        logger.info(f"💰 SELL {quantity} {ticker} ${strike}{type_label} @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_sell_option_limit(
                positionEffect='close',
                creditOrDebit='credit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType=option_type,  # NOW CORRECTLY PASSES call OR put
                timeInForce='gfd'
            )
            return result or {}
        except Exception as e:
            logger.error(f"Sell error: {e}")
            self.circuit_breaker.record_reject()
            return {'error': str(e)}
    
    def get_fill_price(self, order_id: str, max_wait: int = 30) -> Optional[float]:
        for _ in range(max_wait):
            try:
                order = self.rh.orders.get_option_order_info(order_id)
                if order:
                    state = order.get('state', '')
                    if state == 'filled':
                        avg_price = order.get('average_price')
                        if avg_price:
                            self.circuit_breaker.record_fill()
                            return float(avg_price)
                        price = order.get('price')
                        if price:
                            self.circuit_breaker.record_fill()
                            return float(price)
                    elif state in ['cancelled', 'rejected', 'failed']:
                        self.circuit_breaker.record_reject()
                        return None
            except:
                pass
            time.sleep(1)
        return None
    
    def cancel_order(self, order_id: str):
        try:
            self.rh.orders.cancel_option_order(order_id)
        except:
            pass


# =============================================================================
# SCANNER - NOW SCANS BOTH CALLS AND PUTS
# =============================================================================

def scan_for_signals(trader: LaurenTrader, macro: Dict) -> List[Dict]:
    """
    Scan for signals:
    - LONG signals: Underpriced CALLS (score < -0.30)
    - SHORT signals: Buy PUTS when underlying looks bearish (score > 0.40)
    """
    signals = []
    
    # Get all options (calls and puts)
    all_opts = trader.get_all_options()
    
    for o in all_opts:
        ticker = o['ticker']
        spot = o['spot']
        strike = o['strike']
        mid = o['mid']
        bid = o['bid']
        ask = o['ask']
        dte = o['dte']
        oi = o['oi']
        option_type = o['option_type']
        
        if mid < MIN_PREMIUM or bid < MIN_BID:
            continue
        
        if oi < MIN_OI:
            continue
        
        spread_pct = (ask - bid) / mid if mid > 0 else 999
        if spread_pct > MAX_SPREAD_PCT:
            continue
        
        # OTM calculation differs for calls vs puts
        if option_type == 'call':
            otm = (strike - spot) / spot  # Call is OTM when strike > spot
        else:
            otm = (spot - strike) / spot  # Put is OTM when strike < spot
        
        if otm < MIN_OTM or otm > MAX_OTM:
            continue
        
        hv = get_dynamic_vol(trader.rh, ticker)
        T = dte / 365
        
        # Score based on option type
        if option_type == 'call':
            raw_score, iv = quantum_score_call(mid, spot, strike, T, hv)
        else:
            raw_score, iv = quantum_score_put(mid, spot, strike, T, hv)
        
        if raw_score is None:
            continue
        
        # Determine direction based on score AND option type
        direction = None
        
        if option_type == 'call' and raw_score < LONG_THRESH:
            # Underpriced call = LONG signal (buy call, expect up move)
            direction = 'LONG'
        elif option_type == 'put' and raw_score < LONG_THRESH:
            # Underpriced put = SHORT signal (buy put, expect down move)
            direction = 'SHORT'
        
        if direction is None:
            continue
        
        base_conf = score_to_confidence(raw_score)
        
        # Macro adjustment
        if macro and 'trend' in macro:
            if macro['trend'] == 'BULLISH':
                adj = 10 if direction == 'LONG' else -10
            elif macro['trend'] == 'BEARISH':
                adj = -10 if direction == 'LONG' else 10
            else:
                adj = 0
            adj_conf = base_conf + adj
        else:
            adj_conf = base_conf
        
        if adj_conf < MIN_CONFIDENCE:
            continue
        
        signals.append({
            'ticker': ticker,
            'strike': strike,
            'exp': o['exp'],
            'dte': dte,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'spot': spot,
            'otm': otm * 100,
            'oi': oi,
            'confidence': adj_conf,
            'direction': direction,
            'option_type': option_type,  # 'call' or 'put'
            'score': raw_score,
            'iv': iv,
            'hv': hv,
            'spread_pct': spread_pct
        })
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


# =============================================================================
# EXIT CHECK
# =============================================================================

def check_exit(pos: Dict, current_bid: float, current_ask: float) -> Tuple[bool, str, float]:
    entry = pos['entry_price']
    tp_price = pos['tp_price']
    sl_price = pos['sl_price']
    
    current_mid = (current_bid + current_ask) / 2
    
    if current_mid >= tp_price:
        exit_price = calculate_exit_fill(current_bid, current_ask, 'target')
        return True, 'TARGET', exit_price
    
    if current_mid <= sl_price:
        exit_price = calculate_exit_fill(current_bid, current_ask, 'stop')
        return True, 'STOP', exit_price
    
    exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
    days_to_exp = (exp - datetime.now().date()).days
    if days_to_exp <= TIME_STOP_DTE:
        exit_price = calculate_exit_fill(current_bid, current_ask, 'time')
        return True, 'TIME', exit_price
    
    return False, '', 0


# =============================================================================
# MAIN LOOP
# =============================================================================

def run():
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
║     v9 - One Trade Per Day + Proper Shorting                                 ║
║     Position: 100% capital | Max Loss: 35% of position                       ║
║     LONG → Buy Calls | SHORT → Buy Puts                                      ║
║     TP: +50% | SL: -35% | One Trade Per Day                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    trader = LaurenTrader()
    
    if not trader.login():
        logger.error("Login failed!")
        return
    
    trader.update_buying_power()
    logger.info(f"💰 Buying Power: ${trader.buying_power:.2f}")
    logger.info(f"📊 Position Size: 100% | Max Loss: 35% of position")
    logger.info(f"🎯 One trade per day mode")
    
    while True:
        try:
            now = datetime.now()
            state = get_state()
            
            # Check if done for day
            if state.get('done_for_day'):
                logger.info("Done for day - waiting for tomorrow")
                time.sleep(3600)
                continue
            
            # Check daily profit target
            if state['realized_pnl'] >= MAX_DAILY_PROFIT:
                logger.info(f"🎉 Daily profit target: ${state['realized_pnl']:.2f}")
                update_state(done_for_day=True)
                continue
            
            # Check circuit breaker
            safe, reason = trader.circuit_breaker.is_safe_to_trade()
            if not safe:
                logger.warning(f"⚠️ {reason}")
                time.sleep(60)
                continue
            
            # Weekend check
            weekday = now.weekday()
            if weekday >= 5:
                time.sleep(3600)
                continue
            
            # Pre-market check
            hour = now.hour
            minute = now.minute
            
            if hour < 9 or (hour == 9 and minute < 30):
                time.sleep(60)
                continue
            
            # After hours check
            if hour >= 16:
                trader.circuit_breaker.reset()
                time.sleep(3600)
                continue
            
            positions = get_positions()
            
            # =====================================================================
            # EXIT MANAGEMENT - Always check exits first
            # =====================================================================
            
            if positions:
                all_opts = trader.get_all_options()
                
                for pos in positions[:]:
                    current_bid = None
                    current_ask = None
                    
                    # Find current quote for this position
                    for o in all_opts:
                        if (o['ticker'] == pos['ticker'] and
                            o['strike'] == pos['strike'] and
                            o['exp'] == pos['exp'] and
                            o['option_type'] == pos.get('option_type', 'call')):
                            current_bid = o['bid']
                            current_ask = o['ask']
                            break
                    
                    if current_bid is None:
                        exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
                        if now.date() >= exp:
                            # Expired worthless
                            pnl_dollar = -pos['cost']
                            logger.info(f"💸 EXPIRY: {pos['ticker']} | ${pnl_dollar:.2f}")
                            
                            positions.remove(pos)
                            save_positions(positions)
                            
                            state = get_state()
                            update_state(
                                realized_pnl=state['realized_pnl'] + pnl_dollar,
                                losses=state['losses'] + 1,
                                trades_today=state['trades_today'] + 1
                            )
                            log_trade({
                                'date': now.isoformat(),
                                'ticker': pos['ticker'],
                                'strike': pos['strike'],
                                'direction': pos['direction'],
                                'option_type': pos.get('option_type', 'call'),
                                'entry': pos['entry_price'],
                                'exit': 0,
                                'contracts': pos['contracts'],
                                'pnl_pct': -100,
                                'pnl_dollar': pnl_dollar,
                                'reason': 'EXPIRY'
                            })
                        continue
                    
                    mid = (current_bid + current_ask) / 2
                    safe, reason = trader.circuit_breaker.check_spread(current_bid, current_ask, mid)
                    if not safe:
                        continue
                    
                    should_exit, exit_reason, exit_price = check_exit(pos, current_bid, current_ask)
                    
                    if should_exit:
                        opt_label = 'C' if pos.get('option_type', 'call') == 'call' else 'P'
                        logger.info(f"")
                        logger.info(f"{'='*60}")
                        logger.info(f"📤 EXIT: {exit_reason} | {pos['ticker']} ${pos['strike']}{opt_label}")
                        
                        order_result = trader.sell_option(
                            pos['ticker'], pos['strike'], pos['exp'],
                            pos['contracts'], exit_price,
                            pos.get('option_type', 'call')
                        )
                        
                        order_id = order_result.get('id')
                        if not order_id:
                            continue
                        
                        actual_fill = trader.get_fill_price(order_id, max_wait=20)
                        
                        if actual_fill:
                            pnl_pct = (actual_fill - pos['entry_price']) / pos['entry_price'] * 100
                            pnl_dollar = (actual_fill - pos['entry_price']) * pos['contracts'] * 100
                            
                            emoji = "✅" if pnl_dollar > 0 else "❌"
                            logger.info(f"   Fill: ${actual_fill:.2f}")
                            logger.info(f"   P&L: {pnl_pct:+.1f}% | ${pnl_dollar:+.2f} {emoji}")
                            
                            positions.remove(pos)
                            save_positions(positions)
                            
                            state = get_state()
                            if pnl_dollar > 0:
                                update_state(
                                    realized_pnl=state['realized_pnl'] + pnl_dollar,
                                    wins=state['wins'] + 1,
                                    trades_today=state['trades_today'] + 1,
                                    done_for_day=True  # Done after closing a trade
                                )
                            else:
                                update_state(
                                    realized_pnl=state['realized_pnl'] + pnl_dollar,
                                    losses=state['losses'] + 1,
                                    trades_today=state['trades_today'] + 1,
                                    done_for_day=True  # Done after closing a trade
                                )
                            
                            log_trade({
                                'date': now.isoformat(),
                                'ticker': pos['ticker'],
                                'strike': pos['strike'],
                                'direction': pos['direction'],
                                'option_type': pos.get('option_type', 'call'),
                                'entry': pos['entry_price'],
                                'exit': actual_fill,
                                'contracts': pos['contracts'],
                                'pnl_pct': pnl_pct,
                                'pnl_dollar': pnl_dollar,
                                'reason': exit_reason
                            })
                        else:
                            trader.cancel_order(order_id)
            
            # =====================================================================
            # ENTRY - Only if no position AND haven't traded today
            # =====================================================================
            
            positions = get_positions()
            state = get_state()
            
            # ONE TRADE PER DAY: Only look for entries if:
            # 1. No open positions
            # 2. Haven't traded today yet
            if len(positions) == 0 and not state.get('traded_today', False):
                macro = {'trend': 'NEUTRAL'}
                signals = scan_for_signals(trader, macro)
                
                if signals:
                    best = signals[0]
                    
                    safe, reason = trader.circuit_breaker.check_spread(
                        best['bid'], best['ask'], best['mid']
                    )
                    if not safe:
                        logger.warning(f"⚠️ {reason}")
                    else:
                        opt_label = 'C' if best['option_type'] == 'call' else 'P'
                        logger.info(f"")
                        logger.info(f"{'='*60}")
                        logger.info(f"📊 SIGNAL: {best['direction']} {best['ticker']} ${best['strike']}{opt_label}")
                        logger.info(f"   Type: {'CALL' if best['option_type'] == 'call' else 'PUT'}")
                        logger.info(f"   Conf: {best['confidence']:.0f}% | OI: {best['oi']}")
                        
                        entry_price, reasoning = calculate_entry_fill(
                            best['mid'], best['bid'], best['ask'],
                            best['confidence'], best['dte'],
                            best['oi'], best['spread_pct']
                        )
                        
                        trader.update_buying_power()
                        contracts = calculate_position_size(trader.buying_power, entry_price)
                        
                        if contracts <= 0:
                            logger.warning(f"   ⚠️ Cannot size (BP: ${trader.buying_power:.2f})")
                        else:
                            cost = contracts * entry_price * 100
                            max_loss = cost * MAX_LOSS_PCT
                            
                            logger.info(f"   Entry: ${entry_price:.2f} ({reasoning})")
                            logger.info(f"   Size: {contracts} cts = ${cost:.2f}")
                            logger.info(f"   Max Loss (35%): ${max_loss:.2f}")
                            
                            order_result = trader.buy_option(
                                best['ticker'], best['strike'], best['exp'],
                                contracts, entry_price, best['option_type']
                            )
                            
                            order_id = order_result.get('id')
                            if order_id:
                                actual_fill = trader.get_fill_price(order_id, max_wait=15)
                                
                                if actual_fill:
                                    if actual_fill > entry_price * 1.03:
                                        logger.error(f"   ❌ Fill too high: ${actual_fill:.2f}")
                                        trader.cancel_order(order_id)
                                    else:
                                        logger.info(f"   ✅ FILLED @ ${actual_fill:.2f}")
                                        
                                        pos = create_position(
                                            best['ticker'], best['strike'], best['exp'],
                                            actual_fill, contracts, best['confidence'],
                                            best['direction'], best['option_type']
                                        )
                                        
                                        logger.info(f"   TP: ${pos['tp_price']:.2f} | SL: ${pos['sl_price']:.2f}")
                                        logger.info(f"   🎯 Trade placed - done scanning for today")
                                        
                                        positions.append(pos)
                                        save_positions(positions)
                                        
                                        # Mark that we've traded today
                                        update_state(traded_today=True)
                                else:
                                    trader.cancel_order(order_id)
            
            elif state.get('traded_today', False) and len(positions) == 0:
                logger.info("Already traded today - waiting for tomorrow")
            
            logger.info(f"")
            logger.info(f"Next scan in {SCAN_INTERVAL // 60}m...")
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("Stopped")
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
            w = "W" if t['pnl_dollar'] > 0 else "L"
            opt_type = t.get('option_type', 'call')
            opt_label = 'C' if opt_type == 'call' else 'P'
            print(f"{t['date'][:10]} | {t['ticker']} ${t['strike']}{opt_label} | "
                  f"${t['entry']:.2f}→${t['exit']:.2f} | "
                  f"{t['pnl_pct']:+.0f}% | ${t['pnl_dollar']:+.2f} {w}")
        if trades:
            wins = len([t for t in trades if t['pnl_dollar'] > 0])
            total = sum(t['pnl_dollar'] for t in trades)
            print(f"\n{wins}/{len(trades)} wins | ${total:+.2f}")
    
    elif '--clear' in sys.argv:
        for f in [POS_FILE, STATE_FILE, VOL_CACHE_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Cleared")
    
    else:
        run()
