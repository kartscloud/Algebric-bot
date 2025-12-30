"""
================================================================================
MAIN TRADING LOOP
================================================================================

The core trading system that integrates all components:
1. Robinhood client for data and execution
2. Black-Scholes pricing for mispricing detection
3. Graph Laplacian for correlation-based signals
4. HMM regime detection for risk adjustment
5. Risk manager for position sizing and exits

Runs every minute during market hours.

================================================================================
"""

import os
import sys
import time
import signal as sig
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import traceback

# Import all components
from config import (
    print_config, 
    ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD, PAPER_TRADING,
    ACCOUNT_SIZE, DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT,
    PRIMARY_SYMBOLS, SECONDARY_SYMBOLS, ALL_SYMBOLS,
    MIN_DTE, MAX_DTE, MIN_PREMIUM, MAX_PREMIUM,
    MIN_DELTA, MAX_DELTA, MIN_OPEN_INTEREST, MIN_VOLUME, MAX_SPREAD_PCT,
    MIN_IV, MAX_IV, MIN_SIGNAL_SCORE,
    POLL_INTERVAL_SECONDS, AVOID_FIRST_MINUTES, AVOID_LAST_MINUTES,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE, MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    DATA_DIR, LOG_DIR,
    OptionType, OptionQuote, Position, Signal, DailyStats, ExitReason,
    MISPRICING_WEIGHT, IV_WEIGHT, MOMENTUM_WEIGHT, GRAPH_WEIGHT, REGIME_WEIGHT
)

from robinhood import RobinhoodClient
from pricing import BlackScholes, MispricingDetector
from signals import GraphLaplacianSignal, MomentumSignal
from regime import RegimeDetector
from risk import RiskManager, PositionManager

# Setup logging
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{LOG_DIR}/trading.log')
    ]
)
logger = logging.getLogger(__name__)


class QuantOptionsTrader:
    """
    Main trading system integrating all quantitative components
    """
    
    def __init__(self):
        # Core components
        self.client = RobinhoodClient()
        self.mispricing = MispricingDetector()
        self.graph_signal = GraphLaplacianSignal()
        self.momentum_signal = MomentumSignal()
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager(self.regime_detector)
        
        # State
        self.running = False
        self.portfolio_value = ACCOUNT_SIZE
        self.buying_power = ACCOUNT_SIZE
        
        # Data caches
        self.price_cache: Dict[str, float] = {}
        self.options_cache: Dict[str, List[OptionQuote]] = {}
        
        # Logging
        self.trade_log: List[Dict] = []
        self.signal_log: List[Dict] = []
        
    # =========================================================================
    # STARTUP / SHUTDOWN
    # =========================================================================
    
    def start(self):
        """Start the trading system"""
        print_config()
        
        logger.info("=" * 70)
        logger.info("STARTING QUANTITATIVE OPTIONS TRADING SYSTEM")
        logger.info("=" * 70)
        
        # Login to Robinhood
        if not self.client.login(ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD):
            logger.error("Failed to login to Robinhood!")
            logger.error("Check your credentials in config.py")
            return False
        
        # Get initial account info
        self._update_account()
        
        logger.info("")
        logger.info(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Buying Power: ${self.buying_power:,.2f}")
        logger.info(f"Mode: {'LIVE TRADING' if not PAPER_TRADING else 'PAPER TRADING'}")
        logger.info("")
        
        # Initialize risk manager
        self.risk_manager.initialize_day(self.portfolio_value)
        
        # Load existing positions
        self._sync_positions()
        
        # Setup signal handlers
        sig.signal(sig.SIGINT, self._handle_shutdown)
        sig.signal(sig.SIGTERM, self._handle_shutdown)
        
        # Start main loop
        self.running = True
        self._main_loop()
        
        return True
    
    def stop(self):
        """Stop the trading system"""
        logger.info("Stopping trading system...")
        self.running = False
        self._save_state()
        self.client.logout()
        logger.info("Shutdown complete")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal"""
        logger.info("Shutdown signal received")
        self.stop()
        sys.exit(0)
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    def _main_loop(self):
        """Main trading loop - runs every minute"""
        logger.info("Starting main trading loop...")
        logger.info(f"Poll interval: {POLL_INTERVAL_SECONDS} seconds")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Check if we should trade
                if not self._is_market_hours():
                    self._log_waiting("Market closed")
                    time.sleep(60)
                    continue
                
                if self._is_avoid_period():
                    self._log_waiting("Avoid period (open/close)")
                    time.sleep(30)
                    continue
                
                # Check daily limits
                daily_check = self.risk_manager.check_daily_limits()
                if daily_check:
                    logger.info(f"Daily limit: {daily_check}")
                    self._wait_for_next_day()
                    continue
                
                # Run trading cycle
                self._trading_cycle()
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error(traceback.format_exc())
                time.sleep(30)
            
            # Sleep until next cycle
            elapsed = time.time() - loop_start
            sleep_time = max(1, POLL_INTERVAL_SECONDS - elapsed)
            time.sleep(sleep_time)
    
    def _trading_cycle(self):
        """Single trading cycle"""
        cycle_time = datetime.now()
        logger.info("")
        logger.info(f"{'='*60}")
        logger.info(f"CYCLE: {cycle_time.strftime('%H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        # 1. Update account and positions
        self._update_account()
        self._update_positions()
        
        # 2. Fetch market data
        self._fetch_market_data()
        
        # 3. Update regime detection
        self._update_regime()
        
        # 4. Check existing positions for exits
        self._check_exits()
        
        # 5. Generate and evaluate signals
        signals = self._generate_signals()
        
        # 6. Execute best signals
        self._execute_signals(signals)
        
        # 7. Log status
        self._log_status()
    
    # =========================================================================
    # ACCOUNT MANAGEMENT
    # =========================================================================
    
    def _update_account(self):
        """Update account info"""
        try:
            self.portfolio_value = self.client.get_portfolio_value()
            self.buying_power = self.client.get_buying_power()
            
            self.risk_manager.update_portfolio_value(self.portfolio_value)
            self.risk_manager.update_buying_power(self.buying_power)
            
        except Exception as e:
            logger.warning(f"Error updating account: {e}")
    
    def _sync_positions(self):
        """Sync positions from Robinhood"""
        try:
            rh_positions = self.client.get_option_positions()
            logger.info(f"Synced {len(rh_positions)} positions from Robinhood")
        except Exception as e:
            logger.warning(f"Error syncing positions: {e}")
    
    def _update_positions(self):
        """Update current prices for all positions"""
        for position in self.risk_manager.position_manager.get_all_positions():
            try:
                quote = self.client.get_option_quote(
                    position.underlying,
                    position.strike,
                    position.expiration,
                    position.option_type
                )
                if quote:
                    position.update_price(quote.mark)
            except Exception as e:
                logger.debug(f"Error updating position price: {e}")
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def _fetch_market_data(self):
        """Fetch all market data"""
        # Update prices for all symbols
        for symbol in PRIMARY_SYMBOLS:
            try:
                price = self.client.get_price(symbol)
                if price > 0:
                    self.price_cache[symbol] = price
                    
                    # Update signal generators
                    self.graph_signal.update_price(symbol, price)
                    self.momentum_signal.update_price(symbol, price)
                    
                    # Update regime detector
                    if symbol in ['SPY', 'QQQ', 'TQQQ']:
                        if len(self.graph_signal.price_buffer.returns.get(symbol, [])) > 0:
                            ret = self.graph_signal.price_buffer.returns[symbol][-1][1]
                            vol = self.momentum_signal.get_volatility(symbol)
                            self.regime_detector.update(ret, vol)
                    
            except Exception as e:
                logger.debug(f"Error fetching price for {symbol}: {e}")
        
        # Fetch options for top symbols
        symbols_to_scan = PRIMARY_SYMBOLS[:8]  # Top 8
        
        for symbol in symbols_to_scan:
            try:
                self._fetch_options(symbol)
            except Exception as e:
                logger.debug(f"Error fetching options for {symbol}: {e}")
    
    def _fetch_options(self, symbol: str):
        """Fetch options chain for a symbol"""
        expirations = self.client.get_expiration_dates(symbol)
        if not expirations:
            return
        
        # Filter to valid expirations
        today = datetime.now().date()
        valid_exps = []
        
        for exp in expirations[:5]:  # Check first 5
            try:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                dte = (exp_date - today).days
                if MIN_DTE <= dte <= MAX_DTE:
                    valid_exps.append(exp)
            except:
                continue
        
        if not valid_exps:
            return
        
        # Fetch options for nearest 2 expirations
        all_options = []
        for exp in valid_exps[:2]:
            options = self.client.get_options_for_expiration(symbol, exp)
            all_options.extend(options)
        
        self.options_cache[symbol] = all_options
        logger.debug(f"Fetched {len(all_options)} options for {symbol}")
    
    # =========================================================================
    # REGIME DETECTION
    # =========================================================================
    
    def _update_regime(self):
        """Update regime detection model"""
        # Fit model periodically
        if not self.regime_detector.is_fitted:
            self.regime_detector.fit()
        
        # Get current regime
        regime_status = self.regime_detector.get_status()
        logger.info(f"Regime: {regime_status['regime_name']} "
                   f"(pos_mult: {regime_status['position_multiplier']:.2f})")
    
    # =========================================================================
    # EXIT MANAGEMENT
    # =========================================================================
    
    def _check_exits(self):
        """Check all positions for exits"""
        exits = self.risk_manager.get_positions_to_close()
        
        for position, reason, portion in exits:
            self._close_position(position, reason)
    
    def _close_position(self, position: Position, reason: ExitReason):
        """Close a position"""
        logger.info("")
        logger.info(f"{'🎉' if position.unrealized_pnl > 0 else '❌'} CLOSING POSITION")
        logger.info(f"   {position.underlying} {position.strike} {position.option_type}")
        logger.info(f"   Reason: {reason.value}")
        logger.info(f"   P&L: ${position.unrealized_pnl:+.2f} ({position.unrealized_pnl_pct:+.0%})")
        
        # Execute sell order
        result = self.client.sell_option(
            symbol=position.underlying,
            strike=position.strike,
            expiration=position.expiration,
            option_type=position.option_type,
            quantity=position.contracts,
            limit_price=position.current_price * 0.98  # Slightly below to fill
        )
        
        # Record trade
        self.risk_manager.record_trade(position.unrealized_pnl, reason)
        
        # Remove position
        self.risk_manager.position_manager.remove_position(position.position_id)
        
        # Log trade
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'SELL',
            'underlying': position.underlying,
            'strike': position.strike,
            'type': position.option_type,
            'contracts': position.contracts,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'pnl': position.unrealized_pnl,
            'reason': reason.value
        })
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def _generate_signals(self) -> List[Signal]:
        """Generate trading signals for all options"""
        all_signals = []
        
        # Get graph signals for all underlyings
        graph_signals = self.graph_signal.compute_signals(PRIMARY_SYMBOLS)
        
        # Get current regime
        current_regime = self.regime_detector.predict()
        regime_signal = self.regime_detector.get_regime_signal()
        
        for symbol, options in self.options_cache.items():
            if not options:
                continue
            
            # Filter options
            filtered = self._filter_options(options)
            
            # Generate signals for each option
            for opt in filtered:
                signal = self._create_signal(opt, graph_signals, regime_signal, current_regime)
                
                if signal and abs(signal.composite_score) >= MIN_SIGNAL_SCORE:
                    all_signals.append(signal)
                    self.signal_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'underlying': signal.underlying,
                        'strike': signal.strike,
                        'type': signal.option_type.value,
                        'score': signal.composite_score,
                        'mispricing': signal.mispricing_score,
                        'momentum': signal.momentum_score
                    })
        
        # Sort by score
        all_signals.sort(key=lambda s: abs(s.composite_score), reverse=True)
        
        if all_signals:
            logger.info(f"Generated {len(all_signals)} signals")
            for sig in all_signals[:3]:
                logger.info(f"   {sig.underlying} {sig.strike}{sig.option_type.value[0].upper()} "
                           f"score={sig.composite_score:.2f}")
        
        return all_signals
    
    def _filter_options(self, options: List[OptionQuote]) -> List[OptionQuote]:
        """Filter options by criteria"""
        filtered = []
        
        for opt in options:
            # DTE
            if not (MIN_DTE <= opt.dte <= MAX_DTE):
                continue
            
            # Premium
            if not (MIN_PREMIUM <= opt.mark <= MAX_PREMIUM):
                continue
            
            # Delta
            if not (MIN_DELTA <= abs(opt.delta) <= MAX_DELTA):
                continue
            
            # Liquidity
            if opt.open_interest < MIN_OPEN_INTEREST:
                continue
            if opt.volume < MIN_VOLUME:
                continue
            
            # Spread
            if opt.spread_pct > MAX_SPREAD_PCT:
                continue
            
            # IV
            if not (MIN_IV <= opt.implied_volatility <= MAX_IV):
                continue
            
            filtered.append(opt)
        
        return filtered
    
    def _create_signal(
        self,
        opt: OptionQuote,
        graph_signals: Dict[str, float],
        regime_signal: float,
        current_regime: int
    ) -> Optional[Signal]:
        """Create signal for an option"""
        try:
            # Analyze pricing
            pricing = self.mispricing.analyze(opt)
            
            # Get component signals
            mispricing_score = self.mispricing.get_mispricing_signal(pricing.mispricing_pct)
            iv_score = self.mispricing.get_iv_signal(pricing.iv_percentile)
            momentum_score = self.momentum_signal.get_signal(opt.underlying)
            graph_score = graph_signals.get(opt.underlying, 0.0)
            
            # Adjust for option type
            # For CALL: positive signal = bullish = buy
            # For PUT: negative signal = bearish = buy put
            if opt.option_type == OptionType.PUT:
                momentum_score = -momentum_score
                graph_score = -graph_score
            
            # Create signal
            signal = Signal(
                underlying=opt.underlying,
                strike=opt.strike,
                expiration=opt.expiration,
                option_type=opt.option_type,
                mispricing_score=mispricing_score,
                iv_score=iv_score,
                momentum_score=momentum_score,
                graph_score=graph_score,
                regime_score=regime_signal,
                theoretical_price=pricing.theoretical_price,
                market_price=pricing.market_price,
                implied_volatility=pricing.implied_volatility,
                iv_percentile=pricing.iv_percentile,
                current_regime=current_regime,
                delta=pricing.greeks.delta,
                gamma=pricing.greeks.gamma,
                theta=pricing.greeks.theta,
                vega=pricing.greeks.vega
            )
            
            signal.calculate_composite()
            
            return signal
            
        except Exception as e:
            logger.debug(f"Error creating signal: {e}")
            return None
    
    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================
    
    def _execute_signals(self, signals: List[Signal]):
        """Execute best signals"""
        if not signals:
            return
        
        # Only execute positive signals (buy signals)
        buy_signals = [s for s in signals if s.composite_score > 0]
        
        for signal in buy_signals[:2]:  # Max 2 new positions per cycle
            should_trade, reason = self.risk_manager.should_trade_signal(signal)
            
            if not should_trade:
                logger.debug(f"Skipping {signal.underlying}: {reason}")
                continue
            
            # Calculate position size
            contracts = self.risk_manager.calculate_position_size(
                signal,
                signal.market_price,
                self.portfolio_value
            )
            
            if contracts <= 0:
                continue
            
            # Execute trade
            self._open_position(signal, contracts)
    
    def _open_position(self, signal: Signal, contracts: int):
        """Open a new position"""
        logger.info("")
        logger.info(f"🎯 OPENING POSITION")
        logger.info(f"   {signal.underlying} {signal.strike} {signal.option_type.value}")
        logger.info(f"   Contracts: {contracts} @ ${signal.market_price:.2f}")
        logger.info(f"   Signal Score: {signal.composite_score:.2f}")
        logger.info(f"   Components: misp={signal.mispricing_score:.2f}, "
                   f"iv={signal.iv_score:.2f}, mom={signal.momentum_score:.2f}")
        
        # Execute buy order
        result = self.client.buy_option(
            symbol=signal.underlying,
            strike=signal.strike,
            expiration=signal.expiration,
            option_type=signal.option_type.value,
            quantity=contracts,
            limit_price=signal.market_price * 1.02  # Slightly above to fill
        )
        
        # Create position
        position_id = f"{signal.underlying}_{signal.strike}_{signal.option_type.value}_{signal.expiration}"
        
        position = Position(
            position_id=position_id,
            underlying=signal.underlying,
            strike=signal.strike,
            expiration=signal.expiration,
            option_type=signal.option_type.value,
            contracts=contracts,
            entry_price=signal.market_price,
            entry_time=datetime.now(),
            entry_signal_score=signal.composite_score,
            entry_regime=signal.current_regime,
            entry_delta=signal.delta,
            entry_gamma=signal.gamma,
            entry_iv=signal.implied_volatility
        )
        
        # Add to risk manager
        self.risk_manager.position_manager.add_position(position)
        
        # Update buying power estimate
        self.buying_power -= contracts * signal.market_price * 100
        
        # Log trade
        self.trade_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'BUY',
            'underlying': signal.underlying,
            'strike': signal.strike,
            'type': signal.option_type.value,
            'contracts': contracts,
            'price': signal.market_price,
            'signal_score': signal.composite_score
        })
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def _is_market_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        # Weekend
        if now.weekday() >= 5:
            return False
        
        # Market hours
        market_open = now.replace(
            hour=MARKET_OPEN_HOUR, 
            minute=MARKET_OPEN_MINUTE, 
            second=0
        )
        market_close = now.replace(
            hour=MARKET_CLOSE_HOUR, 
            minute=MARKET_CLOSE_MINUTE, 
            second=0
        )
        
        return market_open <= now <= market_close
    
    def _is_avoid_period(self) -> bool:
        """Check if in avoid period"""
        now = datetime.now()
        
        market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE)
        market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE)
        
        # First N minutes
        if now < market_open + timedelta(minutes=AVOID_FIRST_MINUTES):
            return True
        
        # Last N minutes
        if now > market_close - timedelta(minutes=AVOID_LAST_MINUTES):
            return True
        
        return False
    
    def _wait_for_next_day(self):
        """Wait until next trading day"""
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(
            hour=MARKET_OPEN_HOUR, 
            minute=MARKET_OPEN_MINUTE + 5
        )
        
        # Skip weekends
        while tomorrow.weekday() >= 5:
            tomorrow += timedelta(days=1)
        
        logger.info(f"Waiting until {tomorrow.strftime('%Y-%m-%d %H:%M')}")
        
        while datetime.now() < tomorrow and self.running:
            time.sleep(60)
    
    def _log_waiting(self, reason: str):
        """Log waiting status (not every time)"""
        now = datetime.now()
        if now.minute % 5 == 0 and now.second < 5:
            logger.debug(f"Waiting: {reason}")
    
    def _log_status(self):
        """Log current status"""
        status = self.risk_manager.get_status()
        
        logger.info("")
        logger.info(f"Portfolio: ${status['portfolio_value']:,.2f} | "
                   f"Positions: {status['num_positions']} | "
                   f"Exposure: {status['exposure_pct']:.0%}")
        logger.info(f"P&L: Realized ${status['realized_pnl']:+.2f} | "
                   f"Unrealized ${status['unrealized_pnl']:+.2f}")
        logger.info(f"Trades: {status['total_trades']} | "
                   f"Win Rate: {status['win_rate']:.0%}")
    
    def _save_state(self):
        """Save state to disk"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_stats': self.risk_manager.daily_stats.__dict__ if self.risk_manager.daily_stats else None,
            'trade_log': self.trade_log[-100:],
            'signal_log': self.signal_log[-100:]
        }
        
        try:
            with open(f'{DATA_DIR}/state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info("State saved")
        except Exception as e:
            logger.error(f"Error saving state: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗                            ║
║    ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝                            ║
║    ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║                               ║
║    ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║                               ║
║    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║                               ║
║     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝                               ║
║                                                                              ║
║     OPTIONS TRADING SYSTEM                                                   ║
║                                                                              ║
║     Components:                                                              ║
║     • Black-Scholes Mispricing Detection                                     ║
║     • Graph Laplacian Correlation Signals                                    ║
║     • HMM Regime Detection                                                   ║
║     • Multi-Factor Alpha Model                                               ║
║     • Risk-Managed Execution                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    trader = QuantOptionsTrader()
    trader.start()


if __name__ == "__main__":
    main()
