#!/usr/bin/env python3
"""
================================================================================
LAUREN v6 AUTONOMOUS - Auto-Executing Options Trader
================================================================================

Evolved from LAUREN v5 signal scanner → now fully autonomous.

What's new in v6:
- Executes trades automatically (no more SMS signals)
- Coordinates with quant_trader (splits buying power 50/50)
- Separate position/state tracking
- Runs alongside quant_trader without interference

Features:
- MacroScanner: SPY/QQQ/DIA trend analysis
- Black-Scholes mispricing detection (quantum scoring)
- Regime-adjusted confidence
- Auto position management (entries + exits)

Targets: NVDL, TSLL (leveraged single-stock ETFs)
Buying Power: 50% share (other 50% goes to quant_trader)

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

# Historical volatility estimates
HIST_VOL = {
    'NVDL': 0.65,  # 2x NVDA
    'TSLL': 0.90,  # 2x TSLA
}

# Map to underlying for news
NEWS_TICKERS = {
    'NVDL': 'NVDA',
    'TSLL': 'TSLA'
}

# =============================================================================
# BUYING POWER SPLIT
# =============================================================================

BUYING_POWER_SHARE = 0.50  # LAUREN gets 50% of buying power
MAX_POSITION_PCT = 0.40    # 40% of LAUREN's share per position
MAX_POSITIONS = 2          # Max 2 positions at once

# =============================================================================
# SCANNING SETTINGS
# =============================================================================

SCAN_INTERVAL = 180  # 3 minutes between scans
SHORT_THRESH = 0.40  # Score > 0.40 = overpriced = SHORT/SELL CALL
LONG_THRESH = -0.30  # Score < -0.30 = underpriced = LONG/BUY CALL

# =============================================================================
# OPTION FILTERS
# =============================================================================

DTE_MIN = 5          # Min days to expiry
DTE_MAX = 14         # Max days to expiry
MIN_OTM = 0.05       # Min 5% out of the money
MAX_OTM = 0.20       # Max 20% out of the money
MIN_PREMIUM = 0.10   # Min $0.10 option price
MIN_BID = 0.05       # Min $0.05 bid (liquidity)
MIN_OI = 5           # Min open interest

# =============================================================================
# EXIT RULES
# =============================================================================

TARGET = 50          # Take profit at +50%
STOP = -40           # Stop loss at -40%
TIME_STOP_DTE = 1    # Exit if 1 day to expiry

# =============================================================================
# CONFIDENCE
# =============================================================================

MIN_CONFIDENCE = 50  # Min confidence to trade

# =============================================================================
# DAILY LIMITS
# =============================================================================

DAILY_PROFIT_TARGET = 25.0   # Stop when up $25 (LAUREN's share)
DAILY_LOSS_LIMIT = 50.0      # Stop when down $50

# =============================================================================
# FILES
# =============================================================================

DATA_DIR = "lauren_data"
LOG_DIR = "lauren_logs"
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
            'done_for_day': False
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
        self.lauren_buying_power = 0  # LAUREN's 50% share
        
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
                            'oi': int(o.get('open_interest', 0) or 0)
                        })
            except Exception as e:
                logger.debug(f"Error fetching options for {ticker}: {e}")
        
        return opts
    
    def buy_option(self, ticker: str, strike: float, exp: str, 
                   quantity: int, limit_price: float) -> Dict:
        """Buy call option"""
        logger.info(f"🎯 BUYING {quantity} {ticker} ${strike}C @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_buy_option_limit(
                positionEffect='open',
                creditOrDebit='debit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType='call',
                timeInForce='gfd'
            )
            logger.info(f"   Order submitted: {result}")
            return result or {}
        except Exception as e:
            logger.error(f"   Buy error: {e}")
            return {'error': str(e)}
    
    def sell_option(self, ticker: str, strike: float, exp: str,
                    quantity: int, limit_price: float) -> Dict:
        """Sell call option"""
        logger.info(f"💰 SELLING {quantity} {ticker} ${strike}C @ ${limit_price:.2f}")
        
        try:
            result = self.rh.orders.order_sell_option_limit(
                positionEffect='close',
                creditOrDebit='credit',
                price=round(limit_price, 2),
                symbol=ticker,
                quantity=quantity,
                expirationDate=exp,
                strike=strike,
                optionType='call',
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

def scan_for_signals(opts: List[Dict], spots: Dict[str, float], macro: Dict) -> List[Dict]:
    """Scan for trading signals"""
    signals = []
    
    for o in opts:
        ticker = o['ticker']
        spot = spots.get(ticker)
        if not spot:
            continue
        
        strike = o['strike']
        mid = o['mid']
        bid = o['bid']
        dte = o['dte']
        oi = o['oi']
        
        # Filters
        if mid < MIN_PREMIUM or bid < MIN_BID or oi < MIN_OI:
            continue
        
        otm = (strike - spot) / spot
        if otm < MIN_OTM or otm > MAX_OTM:
            continue
        
        T = dte / 365
        hv = HIST_VOL.get(ticker, 0.65)
        raw_score, iv = quantum_score(mid, spot, strike, T, hv)
        
        if raw_score is None:
            continue
        
        # Direction (LAUREN only does LONG for simplicity in auto-execution)
        if raw_score < LONG_THRESH:
            direction = 'LONG'
        else:
            continue  # Skip SHORT signals for autonomous version
        
        # Confidence
        base_conf = score_to_confidence(raw_score)
        adj_conf, adj_reason = adjust_confidence(base_conf, direction, macro)
        
        if adj_conf < MIN_CONFIDENCE:
            continue
        
        signals.append({
            'ticker': ticker,
            'strike': strike,
            'exp': o['exp'],
            'dte': dte,
            'bid': bid,
            'ask': o['ask'],
            'mid': mid,
            'spot': spot,
            'otm': otm * 100,
            'confidence': adj_conf,
            'adj_reason': adj_reason,
            'direction': direction,
            'score': raw_score
        })
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


def check_exit(pos: Dict, opts: List[Dict]) -> Tuple[bool, str, float, float]:
    """Check if position should exit"""
    curr = None
    for o in opts:
        if (o['ticker'] == pos['ticker'] and 
            o['strike'] == pos['strike'] and 
            o['exp'] == pos['exp']):
            curr = o['mid']
            break
    
    if curr is None:
        exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
        if datetime.now().date() >= exp:
            return True, 'EXPIRY', -100, 0
        return False, '', 0, 0
    
    entry = pos['entry']
    pnl = (curr - entry) / entry * 100
    
    if pnl >= TARGET:
        return True, 'TARGET', pnl, curr
    if pnl <= STOP:
        return True, 'STOP', pnl, curr
    
    exp = datetime.strptime(pos['exp'], '%Y-%m-%d').date()
    if (exp - datetime.now().date()).days <= TIME_STOP_DTE:
        return True, 'TIME_STOP', pnl, curr
    
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
║     v6 AUTONOMOUS - Auto-Executing Options Trader                            ║
║     Targets: NVDL, TSLL                                                      ║
║     Buying Power: 50% share                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    trader = LaurenTrader()
    
    if not trader.login():
        logger.error("Failed to login!")
        return
    
    trader.update_buying_power()
    logger.info(f"Total Buying Power: ${trader.buying_power:.2f}")
    logger.info(f"LAUREN's Share (50%): ${trader.lauren_buying_power:.2f}")
    logger.info(f"Tickers: {', '.join(TICKERS)}")
    
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
                    emoji = "💰" if pnl > 0 else "📉"
                    logger.info(f"{emoji} CLOSING: {pos['ticker']} ${pos['strike']}C | {reason} | {pnl:+.0f}%")
                    
                    # Execute sell
                    trader.sell_option(
                        pos['ticker'],
                        pos['strike'],
                        pos['exp'],
                        pos['contracts'],
                        curr * 0.98  # Slightly below to fill
                    )
                    
                    # Calculate actual P&L
                    actual_pnl = (curr - pos['entry']) * pos['contracts'] * 100
                    
                    # Update state
                    state = get_state()
                    new_realized = state['realized_pnl'] + actual_pnl
                    new_trades = state['trades_today'] + 1
                    new_wins = state['wins'] + (1 if pnl > 0 else 0)
                    new_losses = state['losses'] + (1 if pnl <= 0 else 0)
                    update_state(
                        realized_pnl=new_realized,
                        trades_today=new_trades,
                        wins=new_wins,
                        losses=new_losses
                    )
                    
                    # Log trade
                    log_trade({
                        'date': now.strftime('%Y-%m-%d %H:%M'),
                        'ticker': pos['ticker'],
                        'strike': pos['strike'],
                        'direction': 'LONG',
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
                        logger.info(f"   Position: {pos['ticker']} ${pos['strike']}C | {pnl:+.0f}%")
            
            save_positions(new_positions)
            positions = new_positions
            
            # === SCAN FOR NEW ENTRIES ===
            if len(positions) < MAX_POSITIONS:
                signals = scan_for_signals(opts, spots, macro)
                
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
                        contract_cost = best['ask'] * 100
                        contracts = max(1, int(max_spend / contract_cost))
                        
                        # Cap at available buying power
                        total_cost = contracts * contract_cost
                        if total_cost > trader.lauren_buying_power * 0.95:
                            contracts = max(1, int(trader.lauren_buying_power * 0.95 / contract_cost))
                        
                        if contracts >= 1 and contract_cost <= trader.lauren_buying_power:
                            logger.info(f"")
                            logger.info(f"🎯 SIGNAL: {best['ticker']} ${best['strike']}C")
                            logger.info(f"   Confidence: {best['confidence']:.0f}% | {best['adj_reason']}")
                            logger.info(f"   Price: ${best['mid']:.2f} | OTM: {best['otm']:.1f}%")
                            logger.info(f"   Buying {contracts} contracts @ ${best['ask']:.2f}")
                            
                            # Execute buy
                            trader.buy_option(
                                best['ticker'],
                                best['strike'],
                                best['exp'],
                                contracts,
                                best['ask']
                            )
                            
                            # Save position
                            positions.append({
                                'ticker': best['ticker'],
                                'strike': best['strike'],
                                'exp': best['exp'],
                                'entry': best['mid'],
                                'contracts': contracts,
                                'confidence': best['confidence'],
                                'entry_time': now.isoformat()
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
