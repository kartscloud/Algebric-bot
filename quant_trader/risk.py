"""
================================================================================
RISK MANAGER
================================================================================

Comprehensive risk management:
- Position sizing (Kelly criterion, regime-adjusted)
- Portfolio Greeks management
- Stop losses (fixed, trailing, time-based)
- Profit targets (scalp, main, big win)
- Daily P&L limits
- Exposure limits

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from config import (
    Position, Signal, DailyStats, OptionType, ExitReason,
    ACCOUNT_SIZE, MAX_POSITION_PCT, MAX_POSITIONS, MAX_EXPOSURE_PCT,
    MIN_TRADE_SIZE, DAILY_PROFIT_TARGET, DAILY_LOSS_LIMIT,
    SCALP_TARGET_PCT, PROFIT_TARGET_PCT, BIG_WIN_TARGET_PCT,
    STOP_LOSS_PCT, TRAILING_STOP_PCT, TIGHT_STOP_PCT,
    TIME_STOP_MINUTES, MIN_HOLD_FOR_SCALP,
    MIN_SIGNAL_SCORE
)

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages open positions and tracks P&L
    """
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
    def add_position(self, position: Position):
        """Add new position"""
        self.positions[position.position_id] = position
        logger.info(f"Added position: {position.position_id}")
        
    def remove_position(self, position_id: str) -> Optional[Position]:
        """Remove and return position"""
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            self.closed_positions.append(position)
            logger.info(f"Removed position: {position_id}")
            return position
        return None
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)
    
    def get_position_by_option(
        self,
        underlying: str,
        strike: float,
        expiration: str,
        option_type: str
    ) -> Optional[Position]:
        """Find position by option details"""
        for pos in self.positions.values():
            if (pos.underlying == underlying and
                abs(pos.strike - strike) < 0.01 and
                pos.expiration == expiration and
                pos.option_type == option_type):
                return pos
        return None
    
    def update_position_price(self, position_id: str, price: float):
        """Update current price for position"""
        if position_id in self.positions:
            self.positions[position_id].update_price(price)
    
    @property
    def num_positions(self) -> int:
        return len(self.positions)
    
    @property
    def total_exposure(self) -> float:
        """Total market value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def total_cost_basis(self) -> float:
        """Total cost basis of all positions"""
        return sum(pos.cost_basis for pos in self.positions.values())
    
    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    @property
    def portfolio_delta(self) -> float:
        """Net portfolio delta"""
        return sum(
            pos.entry_delta * pos.contracts * 100
            for pos in self.positions.values()
        )
    
    @property
    def portfolio_gamma(self) -> float:
        """Net portfolio gamma"""
        return sum(
            pos.entry_gamma * pos.contracts * 100
            for pos in self.positions.values()
        )
    
    def get_exposure_by_underlying(self) -> Dict[str, float]:
        """Get exposure grouped by underlying"""
        exposure = {}
        for pos in self.positions.values():
            if pos.underlying not in exposure:
                exposure[pos.underlying] = 0
            exposure[pos.underlying] += pos.market_value
        return exposure
    
    def get_all_positions(self) -> List[Position]:
        """Get list of all positions"""
        return list(self.positions.values())


class RiskManager:
    """
    Comprehensive risk management
    """
    
    def __init__(self, regime_detector=None):
        self.position_manager = PositionManager()
        self.regime_detector = regime_detector
        
        # Daily stats
        self.daily_stats: Optional[DailyStats] = None
        self.portfolio_value = ACCOUNT_SIZE
        self.buying_power = ACCOUNT_SIZE
        
    def initialize_day(self, portfolio_value: float):
        """Initialize daily stats"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.daily_stats is None or self.daily_stats.date != today:
            self.daily_stats = DailyStats(
                date=today,
                starting_balance=portfolio_value,
                current_balance=portfolio_value
            )
        
        self.portfolio_value = portfolio_value
        
    def update_portfolio_value(self, value: float):
        """Update current portfolio value"""
        self.portfolio_value = value
        if self.daily_stats:
            self.daily_stats.current_balance = value
            
    def update_buying_power(self, buying_power: float):
        """Update buying power"""
        self.buying_power = buying_power
    
    # =========================================================================
    # POSITION SIZING
    # =========================================================================
    
    def calculate_position_size(
        self,
        signal: Signal,
        option_price: float,
        portfolio_value: float = None
    ) -> int:
        """
        Calculate number of contracts to trade
        
        Uses:
        - Signal strength for base allocation
        - Regime adjustment for volatility
        - Kelly criterion for optimal sizing
        - Max position limits
        
        Returns:
        --------
        int : Number of contracts
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
            
        if option_price <= 0:
            return 0
        
        # Base allocation from signal strength
        signal_strength = abs(signal.composite_score)
        base_allocation = MAX_POSITION_PCT * signal_strength
        
        # Confidence adjustment
        base_allocation *= signal.confidence
        
        # Regime adjustment
        if self.regime_detector:
            regime_mult = self.regime_detector.get_position_multiplier()
            base_allocation *= regime_mult
        
        # Kelly criterion adjustment (simplified)
        # f* = (p * b - q) / b where p = win prob, b = odds, q = 1-p
        # Approximate with signal strength
        kelly_fraction = signal_strength * 0.5  # Half Kelly for safety
        base_allocation = min(base_allocation, kelly_fraction)
        
        # Maximum dollar value
        max_position_value = portfolio_value * base_allocation
        max_position_value = min(max_position_value, portfolio_value * MAX_POSITION_PCT)
        
        # Calculate contracts
        contract_cost = option_price * 100
        max_contracts = int(max_position_value / contract_cost)
        
        # Apply additional limits
        max_contracts = self._apply_position_limits(
            max_contracts,
            contract_cost,
            signal.underlying
        )
        
        # Minimum trade size check
        if max_contracts * contract_cost < MIN_TRADE_SIZE:
            return 0
        
        return max(0, max_contracts)
    
    def _apply_position_limits(
        self,
        proposed_contracts: int,
        contract_cost: float,
        underlying: str
    ) -> int:
        """Apply all position limits"""
        contracts = proposed_contracts
        
        # 1. Max positions limit
        if self.position_manager.num_positions >= MAX_POSITIONS:
            logger.debug("Max positions reached")
            return 0
        
        # 2. Total exposure limit
        current_exposure = self.position_manager.total_exposure
        max_total_exposure = self.portfolio_value * MAX_EXPOSURE_PCT
        remaining_exposure = max_total_exposure - current_exposure
        
        if remaining_exposure <= 0:
            logger.debug("Max exposure reached")
            return 0
        
        max_from_exposure = int(remaining_exposure / contract_cost)
        contracts = min(contracts, max_from_exposure)
        
        # 3. Per-underlying limit (max 50% in one underlying)
        underlying_exposure = self.position_manager.get_exposure_by_underlying()
        current_underlying = underlying_exposure.get(underlying, 0)
        max_underlying = self.portfolio_value * 0.50
        remaining_underlying = max_underlying - current_underlying
        
        if remaining_underlying <= 0:
            logger.debug(f"Max exposure for {underlying} reached")
            return 0
        
        max_from_underlying = int(remaining_underlying / contract_cost)
        contracts = min(contracts, max_from_underlying)
        
        # 4. Buying power check
        max_from_bp = int(self.buying_power * 0.95 / contract_cost)
        contracts = min(contracts, max_from_bp)
        
        return max(0, contracts)
    
    # =========================================================================
    # EXIT CHECKING
    # =========================================================================
    
    def check_exit(self, position: Position) -> Optional[Tuple[ExitReason, float]]:
        """
        Check if position should be exited
        
        Returns:
        --------
        Optional[Tuple[ExitReason, float]] : (reason, portion_to_close)
            None if no exit needed
        """
        pnl_pct = position.unrealized_pnl_pct
        hold_time = position.hold_time_minutes
        drawdown = position.drawdown_from_high
        
        # Get regime-adjusted stops
        if self.regime_detector:
            stop_mult = self.regime_detector.get_stop_multiplier()
        else:
            stop_mult = 1.0
        
        adjusted_stop = STOP_LOSS_PCT * stop_mult
        adjusted_trailing = TRAILING_STOP_PCT * stop_mult
        
        # =====================================================================
        # PROFIT TARGETS (check first - take profits!)
        # =====================================================================
        
        # Big win - let it run but take some off
        if pnl_pct >= BIG_WIN_TARGET_PCT:
            logger.info(f"🎉 BIG WIN: {position.underlying} +{pnl_pct:.0%}")
            return (ExitReason.BIG_WIN, 1.0)
        
        # Main profit target
        if pnl_pct >= PROFIT_TARGET_PCT:
            logger.info(f"✅ PROFIT TARGET: {position.underlying} +{pnl_pct:.0%}")
            return (ExitReason.PROFIT_TARGET, 1.0)
        
        # Scalp target (quick profit after minimum hold)
        if pnl_pct >= SCALP_TARGET_PCT and hold_time >= MIN_HOLD_FOR_SCALP:
            logger.info(f"💰 SCALP: {position.underlying} +{pnl_pct:.0%} ({hold_time:.0f}min)")
            return (ExitReason.SCALP, 1.0)
        
        # =====================================================================
        # STOP LOSSES
        # =====================================================================
        
        # Hard stop loss
        if pnl_pct <= -adjusted_stop:
            logger.warning(f"🛑 STOP LOSS: {position.underlying} {pnl_pct:.0%}")
            return (ExitReason.STOP_LOSS, 1.0)
        
        # Trailing stop (only if we've been profitable)
        if position.high_price > position.entry_price * 1.05:  # Was up 5%+
            if drawdown <= -adjusted_trailing:
                logger.warning(f"📉 TRAILING STOP: {position.underlying} {drawdown:.0%} from high")
                return (ExitReason.TRAILING_STOP, 1.0)
        
        # Tight stop for scalps (if holding for quick profit)
        if hold_time < MIN_HOLD_FOR_SCALP and pnl_pct <= -TIGHT_STOP_PCT:
            logger.warning(f"⚡ TIGHT STOP: {position.underlying} {pnl_pct:.0%}")
            return (ExitReason.STOP_LOSS, 1.0)
        
        # =====================================================================
        # TIME STOP
        # =====================================================================
        
        # Exit flat positions after time limit
        if hold_time >= TIME_STOP_MINUTES and abs(pnl_pct) < 0.05:
            logger.info(f"⏰ TIME STOP: {position.underlying} {hold_time:.0f}min, {pnl_pct:.0%}")
            return (ExitReason.TIME_STOP, 1.0)
        
        return None
    
    def get_positions_to_close(self) -> List[Tuple[Position, ExitReason, float]]:
        """
        Check all positions for exits
        
        Returns:
        --------
        List of (position, reason, portion)
        """
        to_close = []
        
        for position in self.position_manager.get_all_positions():
            result = self.check_exit(position)
            if result:
                reason, portion = result
                to_close.append((position, reason, portion))
        
        return to_close
    
    # =========================================================================
    # DAILY LIMITS
    # =========================================================================
    
    def check_daily_limits(self) -> Optional[str]:
        """
        Check if daily P&L limits hit
        
        Returns:
        --------
        str : Reason for stopping, or None if OK to continue
        """
        if self.daily_stats is None:
            return None
        
        # Check if new day
        today = datetime.now().strftime('%Y-%m-%d')
        if self.daily_stats.date != today:
            # Reset for new day
            self.daily_stats = DailyStats(
                date=today,
                starting_balance=self.portfolio_value,
                current_balance=self.portfolio_value
            )
            return None
        
        # Calculate total P&L including unrealized
        realized = self.daily_stats.realized_pnl
        unrealized = self.position_manager.total_unrealized_pnl
        total_pnl = realized + unrealized
        
        # Check profit target
        if realized >= DAILY_PROFIT_TARGET:
            return f"Daily profit target reached: +${realized:.2f}"
        
        # Check loss limit
        if realized <= -DAILY_LOSS_LIMIT:
            return f"Daily loss limit reached: -${abs(realized):.2f}"
        
        # Warn if approaching limits
        if realized >= DAILY_PROFIT_TARGET * 0.8:
            logger.info(f"Approaching daily target: ${realized:.2f} / ${DAILY_PROFIT_TARGET:.2f}")
        
        if realized <= -DAILY_LOSS_LIMIT * 0.8:
            logger.warning(f"Approaching daily stop: ${realized:.2f} / -${DAILY_LOSS_LIMIT:.2f}")
        
        return None
    
    # =========================================================================
    # TRADE RECORDING
    # =========================================================================
    
    def record_trade(self, pnl: float, exit_reason: ExitReason):
        """Record completed trade"""
        if self.daily_stats:
            self.daily_stats.record_trade(pnl)
        
        logger.info(f"Trade recorded: ${pnl:+.2f} ({exit_reason.value})")
    
    # =========================================================================
    # SIGNAL FILTERING
    # =========================================================================
    
    def should_trade_signal(self, signal: Signal) -> Tuple[bool, str]:
        """
        Check if a signal should be traded
        
        Returns:
        --------
        (should_trade, reason)
        """
        # Check signal strength
        if abs(signal.composite_score) < MIN_SIGNAL_SCORE:
            return False, f"Signal too weak: {signal.composite_score:.2f}"
        
        # Check if we already have this position
        existing = self.position_manager.get_position_by_option(
            signal.underlying,
            signal.strike,
            signal.expiration,
            signal.option_type.value
        )
        if existing:
            return False, "Position already exists"
        
        # Check daily limits
        daily_check = self.check_daily_limits()
        if daily_check:
            return False, daily_check
        
        # Check position limits
        if self.position_manager.num_positions >= MAX_POSITIONS:
            return False, "Max positions reached"
        
        # Check buying power
        if self.buying_power < MIN_TRADE_SIZE:
            return False, f"Insufficient buying power: ${self.buying_power:.2f}"
        
        return True, "OK"
    
    # =========================================================================
    # STATUS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get comprehensive risk status"""
        return {
            'portfolio_value': self.portfolio_value,
            'buying_power': self.buying_power,
            'num_positions': self.position_manager.num_positions,
            'total_exposure': self.position_manager.total_exposure,
            'exposure_pct': self.position_manager.total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0,
            'unrealized_pnl': self.position_manager.total_unrealized_pnl,
            'realized_pnl': self.daily_stats.realized_pnl if self.daily_stats else 0,
            'total_trades': self.daily_stats.total_trades if self.daily_stats else 0,
            'win_rate': self.daily_stats.win_rate if self.daily_stats else 0,
            'portfolio_delta': self.position_manager.portfolio_delta,
            'portfolio_gamma': self.position_manager.portfolio_gamma,
            'positions': [
                {
                    'underlying': p.underlying,
                    'strike': p.strike,
                    'type': p.option_type,
                    'contracts': p.contracts,
                    'entry': p.entry_price,
                    'current': p.current_price,
                    'pnl': p.unrealized_pnl,
                    'pnl_pct': p.unrealized_pnl_pct,
                    'hold_time': p.hold_time_minutes
                }
                for p in self.position_manager.get_all_positions()
            ]
        }
