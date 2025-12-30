"""
================================================================================
ROBINHOOD API CLIENT
================================================================================

Complete wrapper for Robinhood API operations:
- Authentication
- Market data (quotes, options chains)
- Account info (portfolio, buying power)
- Order execution (buy/sell options)
- Position management

================================================================================
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from config import (
    ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD, PAPER_TRADING,
    OptionType, OptionQuote
)

logger = logging.getLogger(__name__)

# Import robin_stocks
try:
    import robin_stocks.robinhood as rh
    ROBINHOOD_AVAILABLE = True
except ImportError:
    ROBINHOOD_AVAILABLE = False
    logger.error("robin_stocks not installed! Run: pip install robin_stocks")


class RobinhoodClient:
    """
    Complete Robinhood API client
    """
    
    def __init__(self):
        self.authenticated = False
        self.login_time: Optional[datetime] = None
        
        # Cache with TTL
        self._cache: Dict[str, Tuple[any, datetime]] = {}
        self._cache_ttl = 30  # seconds
        
    # =========================================================================
    # AUTHENTICATION
    # =========================================================================
    
    def login(
        self,
        username: str = None,
        password: str = None,
        mfa_code: str = None
    ) -> bool:
        """
        Login to Robinhood
        
        Parameters:
        -----------
        username : str - Email (defaults to config)
        password : str - Password (defaults to config)
        mfa_code : str - 2FA code if enabled
        
        Returns:
        --------
        bool : Success
        """
        if not ROBINHOOD_AVAILABLE:
            logger.error("robin_stocks not available")
            return False
        
        # Use provided or config credentials
        username = username or ROBINHOOD_USERNAME
        password = password or ROBINHOOD_PASSWORD
        
        if not username or not password:
            logger.error("No credentials provided")
            return False
        
        logger.info(f"Logging into Robinhood as {username}...")
        
        try:
            if mfa_code:
                result = rh.login(
                    username, password,
                    mfa_code=mfa_code,
                    store_session=True
                )
            else:
                result = rh.login(
                    username, password,
                    store_session=True
                )
            
            if result:
                self.authenticated = True
                self.login_time = datetime.now()
                logger.info("✅ Login successful")
                return True
            else:
                logger.error("❌ Login failed - no result returned")
                return False
                
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    def logout(self):
        """Logout from Robinhood"""
        if ROBINHOOD_AVAILABLE:
            try:
                rh.logout()
            except:
                pass
        self.authenticated = False
        logger.info("Logged out")
    
    def _check_auth(self):
        """Verify authentication"""
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call login() first.")
    
    # =========================================================================
    # CACHING
    # =========================================================================
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get cached value if not expired"""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return value
        return None
    
    def _set_cached(self, key: str, value: any):
        """Cache a value"""
        self._cache[key] = (value, datetime.now())
    
    def clear_cache(self):
        """Clear all cached data"""
        self._cache.clear()
    
    # =========================================================================
    # ACCOUNT DATA
    # =========================================================================
    
    def get_account(self) -> Dict:
        """Get account profile"""
        self._check_auth()
        try:
            return rh.profiles.load_account_profile() or {}
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}
    
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        self._check_auth()
        try:
            profile = rh.profiles.load_portfolio_profile()
            return float(profile.get('equity', 0))
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    def get_buying_power(self) -> float:
        """Get available buying power"""
        self._check_auth()
        try:
            account = rh.profiles.load_account_profile()
            return float(account.get('buying_power', 0))
        except Exception as e:
            logger.error(f"Error getting buying power: {e}")
            return 0.0
    
    def get_positions(self) -> List[Dict]:
        """Get open stock positions"""
        self._check_auth()
        try:
            return rh.account.get_open_stock_positions() or []
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_option_positions(self) -> List[Dict]:
        """Get open option positions"""
        self._check_auth()
        try:
            return rh.options.get_open_option_positions() or []
        except Exception as e:
            logger.error(f"Error getting option positions: {e}")
            return []
    
    # =========================================================================
    # MARKET DATA - STOCKS
    # =========================================================================
    
    def get_quote(self, symbol: str) -> Dict:
        """Get stock quote"""
        self._check_auth()
        
        # Check cache
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            quote = rh.stocks.get_stock_quote_by_symbol(symbol)
            if quote:
                self._set_cached(cache_key, quote)
            return quote or {}
        except Exception as e:
            logger.warning(f"Error getting quote for {symbol}: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """Get current price"""
        quote = self.get_quote(symbol)
        return float(quote.get('last_trade_price', 0))
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_quote(symbol)
        return quotes
    
    def get_historicals(
        self,
        symbol: str,
        interval: str = '5minute',
        span: str = 'day'
    ) -> List[Dict]:
        """
        Get historical data
        
        Parameters:
        -----------
        interval : '5minute', '10minute', 'hour', 'day', 'week'
        span : 'day', 'week', 'month', '3month', 'year', '5year'
        """
        self._check_auth()
        try:
            return rh.stocks.get_stock_historicals(
                symbol,
                interval=interval,
                span=span
            ) or []
        except Exception as e:
            logger.warning(f"Error getting historicals for {symbol}: {e}")
            return []
    
    # =========================================================================
    # MARKET DATA - OPTIONS
    # =========================================================================
    
    def get_expiration_dates(self, symbol: str) -> List[str]:
        """Get available expiration dates for options"""
        self._check_auth()
        
        cache_key = f"expirations_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            chains = rh.options.get_chains(symbol)
            dates = chains.get('expiration_dates', []) if chains else []
            if dates:
                self._set_cached(cache_key, dates)
            return dates
        except Exception as e:
            logger.warning(f"Error getting expirations for {symbol}: {e}")
            return []
    
    def get_options_for_expiration(
        self,
        symbol: str,
        expiration: str,
        option_type: str = None
    ) -> List[OptionQuote]:
        """
        Get all options for a specific expiration
        
        Parameters:
        -----------
        symbol : str - Underlying symbol
        expiration : str - Expiration date (YYYY-MM-DD)
        option_type : str - 'call', 'put', or None for both
        
        Returns:
        --------
        List[OptionQuote]
        """
        self._check_auth()
        options = []
        
        # Get underlying price
        underlying_price = self.get_price(symbol)
        if underlying_price <= 0:
            logger.warning(f"Could not get price for {symbol}")
            return []
        
        try:
            # Get options
            if option_type:
                raw_options = rh.options.find_options_by_expiration(
                    symbol,
                    expirationDate=expiration,
                    optionType=option_type
                ) or []
            else:
                calls = rh.options.find_options_by_expiration(
                    symbol, expirationDate=expiration, optionType='call'
                ) or []
                puts = rh.options.find_options_by_expiration(
                    symbol, expirationDate=expiration, optionType='put'
                ) or []
                raw_options = calls + puts
            
            if not raw_options:
                return []
            
            # Get market data
            option_ids = [opt['id'] for opt in raw_options if 'id' in opt]
            
            if option_ids:
                market_data = rh.options.get_option_market_data_by_id(option_ids)
                md_by_id = {}
                for md in (market_data or []):
                    try:
                        opt_id = md['instrument'].split('/')[-2]
                        md_by_id[opt_id] = md
                    except:
                        pass
            else:
                md_by_id = {}
            
            # Build OptionQuote objects
            for opt in raw_options:
                try:
                    opt_id = opt.get('id', '')
                    md = md_by_id.get(opt_id, {})
                    
                    opt_type = OptionType.CALL if opt.get('type') == 'call' else OptionType.PUT
                    
                    quote = OptionQuote(
                        underlying=symbol,
                        underlying_price=underlying_price,
                        strike=float(opt.get('strike_price', 0)),
                        expiration=opt.get('expiration_date', expiration),
                        option_type=opt_type,
                        bid=float(md.get('bid_price', 0) or 0),
                        ask=float(md.get('ask_price', 0) or 0),
                        mark=float(md.get('mark_price', 0) or 0),
                        last=float(md.get('last_trade_price', 0) or 0),
                        volume=int(md.get('volume', 0) or 0),
                        open_interest=int(md.get('open_interest', 0) or 0),
                        delta=float(md.get('delta', 0) or 0),
                        gamma=float(md.get('gamma', 0) or 0),
                        theta=float(md.get('theta', 0) or 0),
                        vega=float(md.get('vega', 0) or 0),
                        rho=float(md.get('rho', 0) or 0),
                        implied_volatility=float(md.get('implied_volatility', 0) or 0),
                        timestamp=datetime.now()
                    )
                    
                    if quote.mark > 0:
                        options.append(quote)
                        
                except Exception as e:
                    logger.debug(f"Error parsing option: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Error getting options for {symbol} {expiration}: {e}")
        
        return options
    
    def get_option_quote(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str
    ) -> Optional[OptionQuote]:
        """Get quote for specific option"""
        options = self.get_options_for_expiration(symbol, expiration, option_type)
        
        for opt in options:
            if abs(opt.strike - strike) < 0.01:
                return opt
        
        return None
    
    # =========================================================================
    # ORDER EXECUTION
    # =========================================================================
    
    def buy_option(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,
        quantity: int,
        limit_price: float = None,
        time_in_force: str = 'gfd'
    ) -> Dict:
        """
        Buy to open option position
        
        Parameters:
        -----------
        symbol : str - Underlying symbol
        strike : float - Strike price
        expiration : str - Expiration (YYYY-MM-DD)
        option_type : str - 'call' or 'put'
        quantity : int - Number of contracts
        limit_price : float - Limit price (if None, uses market)
        time_in_force : str - 'gfd' (good for day) or 'gtc'
        
        Returns:
        --------
        Dict : Order result
        """
        if PAPER_TRADING:
            logger.info(f"[PAPER] BUY {quantity} {symbol} {strike} {option_type} @ {limit_price}")
            return {
                'status': 'paper_trade',
                'filled': True,
                'symbol': symbol,
                'strike': strike,
                'quantity': quantity,
                'price': limit_price
            }
        
        self._check_auth()
        
        try:
            if limit_price:
                result = rh.orders.order_buy_option_limit(
                    positionEffect='open',
                    creditOrDebit='debit',
                    price=round(limit_price, 2),
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration,
                    strike=strike,
                    optionType=option_type,
                    timeInForce=time_in_force
                )
            else:
                # Market order (not recommended for options)
                result = rh.orders.order_buy_option_limit(
                    positionEffect='open',
                    creditOrDebit='debit',
                    price=999.99,  # High limit for market-like fill
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration,
                    strike=strike,
                    optionType=option_type,
                    timeInForce=time_in_force
                )
            
            logger.info(f"Buy order submitted: {result}")
            return result or {}
            
        except Exception as e:
            logger.error(f"Error submitting buy order: {e}")
            return {'error': str(e)}
    
    def sell_option(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,
        quantity: int,
        limit_price: float = None,
        time_in_force: str = 'gfd'
    ) -> Dict:
        """
        Sell to close option position
        """
        if PAPER_TRADING:
            logger.info(f"[PAPER] SELL {quantity} {symbol} {strike} {option_type} @ {limit_price}")
            return {
                'status': 'paper_trade',
                'filled': True,
                'symbol': symbol,
                'strike': strike,
                'quantity': quantity,
                'price': limit_price
            }
        
        self._check_auth()
        
        try:
            if limit_price:
                result = rh.orders.order_sell_option_limit(
                    positionEffect='close',
                    creditOrDebit='credit',
                    price=round(limit_price, 2),
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration,
                    strike=strike,
                    optionType=option_type,
                    timeInForce=time_in_force
                )
            else:
                result = rh.orders.order_sell_option_limit(
                    positionEffect='close',
                    creditOrDebit='credit',
                    price=0.01,  # Low limit for market-like fill
                    symbol=symbol,
                    quantity=quantity,
                    expirationDate=expiration,
                    strike=strike,
                    optionType=option_type,
                    timeInForce=time_in_force
                )
            
            logger.info(f"Sell order submitted: {result}")
            return result or {}
            
        except Exception as e:
            logger.error(f"Error submitting sell order: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order"""
        if PAPER_TRADING:
            return True
        
        self._check_auth()
        
        try:
            rh.orders.cancel_option_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        self._check_auth()
        try:
            return rh.orders.get_all_open_option_orders() or []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
