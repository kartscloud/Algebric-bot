"""
================================================================================
BLACK-SCHOLES PRICING ENGINE
================================================================================

Implements:
- Black-Scholes option pricing
- Full Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation (Brent's method + Newton-Raphson fallback)
- Mispricing detection (theoretical vs market)
- IV percentile calculation

This is the core of the LAUREN-style mispricing detection system.
================================================================================
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import logging

from config import (
    OptionType, OptionQuote, Signal,
    MIN_MISPRICING_PCT, IV_LOOKBACK_MINUTES,
    IV_BUY_THRESHOLD, IV_SELL_THRESHOLD
)

logger = logging.getLogger(__name__)


@dataclass
class Greeks:
    """Container for option Greeks"""
    delta: float
    gamma: float
    theta: float  # Daily theta
    vega: float   # Per 1% IV move
    rho: float    # Per 1% rate move
    
    def to_dict(self) -> Dict:
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho
        }


@dataclass
class PricingResult:
    """Complete pricing analysis result"""
    # Prices
    theoretical_price: float
    market_price: float
    intrinsic_value: float
    time_value: float
    
    # Mispricing
    mispricing: float           # theoretical - market
    mispricing_pct: float       # mispricing / market
    
    # Volatility
    implied_volatility: float
    iv_percentile: float
    
    # Greeks
    greeks: Greeks
    
    @property
    def is_underpriced(self) -> bool:
        """Option is cheaper than theoretical value"""
        return self.mispricing > 0 and self.mispricing_pct > MIN_MISPRICING_PCT
    
    @property
    def is_overpriced(self) -> bool:
        """Option is more expensive than theoretical value"""
        return self.mispricing < 0 and abs(self.mispricing_pct) > MIN_MISPRICING_PCT


class BlackScholes:
    """
    Black-Scholes Option Pricing Model
    
    Formulas:
    ---------
    d1 = (ln(S/K) + (r - q + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T
    
    Call Price = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
    Put Price  = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
    
    Greeks:
    -------
    Delta (∂V/∂S):
        Call: e^(-qT)·N(d1)
        Put:  -e^(-qT)·N(-d1)
    
    Gamma (∂²V/∂S²):
        Both: e^(-qT)·N'(d1) / (S·σ·√T)
    
    Theta (∂V/∂T):
        Call: -S·σ·e^(-qT)·N'(d1)/(2√T) + q·S·e^(-qT)·N(d1) - r·K·e^(-rT)·N(d2)
        Put:  -S·σ·e^(-qT)·N'(d1)/(2√T) - q·S·e^(-qT)·N(-d1) + r·K·e^(-rT)·N(-d2)
    
    Vega (∂V/∂σ):
        Both: S·e^(-qT)·√T·N'(d1)
    
    Rho (∂V/∂r):
        Call: K·T·e^(-rT)·N(d2)
        Put:  -K·T·e^(-rT)·N(-d2)
    """
    
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize Black-Scholes model
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free interest rate (default 5%)
        dividend_yield : float
            Annual dividend yield (default 0%)
        """
        self.r = risk_free_rate
        self.q = dividend_yield
        
    def _validate_inputs(self, S: float, K: float, T: float, sigma: float) -> bool:
        """Validate inputs are reasonable"""
        if S <= 0 or K <= 0:
            return False
        if T < 0:
            return False
        if sigma <= 0:
            return False
        return True
        
    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter"""
        if T <= 0 or sigma <= 0:
            return 0.0
        numerator = np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * T
        denominator = sigma * np.sqrt(T)
        return numerator / denominator
    
    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return self._d1(S, K, T, sigma) - sigma * np.sqrt(T)
    
    def price(
        self,
        S: float,      # Spot price
        K: float,      # Strike price
        T: float,      # Time to expiry in years
        sigma: float,  # Volatility (annualized)
        option_type: OptionType
    ) -> float:
        """
        Calculate theoretical option price
        
        Parameters:
        -----------
        S : float - Current price of underlying
        K : float - Strike price
        T : float - Time to expiration in years (e.g., 7/365 for 7 days)
        sigma : float - Annualized volatility (e.g., 0.30 for 30%)
        option_type : OptionType - CALL or PUT
        
        Returns:
        --------
        float : Theoretical option price
        """
        # At expiration, return intrinsic value
        if T <= 0:
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Input validation
        if not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        if option_type == OptionType.CALL:
            price = (S * np.exp(-self.q * T) * norm.cdf(d1) - 
                    K * np.exp(-self.r * T) * norm.cdf(d2))
        else:  # PUT
            price = (K * np.exp(-self.r * T) * norm.cdf(-d2) - 
                    S * np.exp(-self.q * T) * norm.cdf(-d1))
        
        return max(price, 0.0)
    
    def delta(
        self,
        S: float, K: float, T: float, sigma: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate Delta - rate of change of option price with respect to underlying
        
        Call Delta: 0 to 1 (typically 0.5 ATM)
        Put Delta: -1 to 0 (typically -0.5 ATM)
        """
        if T <= 0:
            # At expiration, delta is 1 or 0 for ITM/OTM
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
                
        if not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d1 = self._d1(S, K, T, sigma)
        
        if option_type == OptionType.CALL:
            return np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return -np.exp(-self.q * T) * norm.cdf(-d1)
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Gamma - rate of change of delta with respect to underlying
        
        Same for calls and puts
        Highest ATM, decreases as option moves ITM/OTM
        """
        if T <= 0 or not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d1 = self._d1(S, K, T, sigma)
        
        return (np.exp(-self.q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    def theta(
        self,
        S: float, K: float, T: float, sigma: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate Theta - rate of time decay
        
        Returns daily theta (divided by 365)
        Usually negative (options lose value over time)
        """
        if T <= 0 or not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        # First term (same for both)
        first_term = -(S * sigma * np.exp(-self.q * T) * norm.pdf(d1)) / (2 * np.sqrt(T))
        
        if option_type == OptionType.CALL:
            theta = (first_term 
                    + self.q * S * np.exp(-self.q * T) * norm.cdf(d1)
                    - self.r * K * np.exp(-self.r * T) * norm.cdf(d2))
        else:
            theta = (first_term
                    - self.q * S * np.exp(-self.q * T) * norm.cdf(-d1)
                    + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2))
        
        # Return daily theta
        return theta / 365
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Vega - sensitivity to volatility
        
        Same for calls and puts
        Returns vega per 1% (0.01) move in volatility
        """
        if T <= 0 or not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d1 = self._d1(S, K, T, sigma)
        
        vega = S * np.exp(-self.q * T) * np.sqrt(T) * norm.pdf(d1)
        
        # Return per 1% vol move
        return vega / 100
    
    def rho(
        self,
        S: float, K: float, T: float, sigma: float,
        option_type: OptionType
    ) -> float:
        """
        Calculate Rho - sensitivity to interest rates
        
        Returns rho per 1% (0.01) move in rates
        """
        if T <= 0 or not self._validate_inputs(S, K, T, sigma):
            return 0.0
            
        d2 = self._d2(S, K, T, sigma)
        
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2)
        
        # Return per 1% rate move
        return rho / 100
    
    def greeks(
        self,
        S: float, K: float, T: float, sigma: float,
        option_type: OptionType
    ) -> Greeks:
        """Calculate all Greeks at once"""
        return Greeks(
            delta=self.delta(S, K, T, sigma, option_type),
            gamma=self.gamma(S, K, T, sigma),
            theta=self.theta(S, K, T, sigma, option_type),
            vega=self.vega(S, K, T, sigma),
            rho=self.rho(S, K, T, sigma, option_type)
        )
    
    def implied_volatility(
        self,
        market_price: float,
        S: float, K: float, T: float,
        option_type: OptionType,
        precision: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calculate implied volatility from market price
        
        Uses Brent's method for root finding with Newton-Raphson fallback
        
        Parameters:
        -----------
        market_price : float - Observed market price
        S, K, T : float - Spot, strike, time to expiry
        option_type : OptionType
        precision : float - Convergence tolerance
        max_iterations : int - Maximum iterations
        
        Returns:
        --------
        float : Implied volatility (annualized)
        """
        if market_price <= 0 or T <= 0:
            return 0.0
        
        # Check intrinsic value
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
            
        # Price below intrinsic is invalid
        if market_price < intrinsic * 0.99:
            return 0.0
        
        # Objective function: find sigma where BS price = market price
        def objective(sigma):
            return self.price(S, K, T, sigma, option_type) - market_price
        
        # Try Brent's method first (most robust)
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=precision, maxiter=max_iterations)
            return iv
        except (ValueError, RuntimeError):
            pass
        
        # Fallback: Newton-Raphson
        sigma = 0.30  # Initial guess
        
        for i in range(max_iterations):
            price = self.price(S, K, T, sigma, option_type)
            vega = self.vega(S, K, T, sigma) * 100  # Undo the /100
            
            if abs(vega) < 1e-10:
                break
            
            diff = market_price - price
            if abs(diff) < precision:
                return sigma
            
            # Newton step
            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 5.0))
        
        return sigma


class MispricingDetector:
    """
    Detects mispriced options using Black-Scholes
    
    Compares theoretical BS price to market price to find:
    - Underpriced options (market < theoretical) -> BUY signal
    - Overpriced options (market > theoretical) -> SELL signal
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.bs = BlackScholes(risk_free_rate=risk_free_rate)
        
        # IV history for percentile calculations
        # Key: symbol, Value: deque of (timestamp, IV) tuples
        self.iv_history: Dict[str, deque] = {}
        
    def analyze(self, quote: OptionQuote) -> PricingResult:
        """
        Analyze an option for mispricing
        
        Parameters:
        -----------
        quote : OptionQuote - Complete option quote data
        
        Returns:
        --------
        PricingResult : Complete pricing analysis
        """
        S = quote.underlying_price
        K = quote.strike
        T = quote.dte / 365  # Convert days to years
        market_price = quote.mark
        option_type = quote.option_type
        
        # Get IV (use provided or calculate)
        if quote.implied_volatility > 0:
            iv = quote.implied_volatility
        else:
            iv = self.bs.implied_volatility(market_price, S, K, T, option_type)
        
        # Calculate theoretical price using IV
        theoretical_price = self.bs.price(S, K, T, iv, option_type)
        
        # Calculate Greeks
        greeks = self.bs.greeks(S, K, T, iv, option_type)
        
        # Mispricing
        mispricing = theoretical_price - market_price
        mispricing_pct = mispricing / market_price if market_price > 0 else 0
        
        # Intrinsic and time value
        if option_type == OptionType.CALL:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)
        time_value = market_price - intrinsic
        
        # IV percentile
        iv_percentile = self.get_iv_percentile(quote.underlying, iv)
        
        # Update IV history
        self.update_iv_history(quote.underlying, iv)
        
        return PricingResult(
            theoretical_price=theoretical_price,
            market_price=market_price,
            intrinsic_value=intrinsic,
            time_value=time_value,
            mispricing=mispricing,
            mispricing_pct=mispricing_pct,
            implied_volatility=iv,
            iv_percentile=iv_percentile,
            greeks=greeks
        )
    
    def update_iv_history(self, symbol: str, iv: float):
        """Add IV observation to history"""
        if symbol not in self.iv_history:
            self.iv_history[symbol] = deque(maxlen=IV_LOOKBACK_MINUTES)
        
        self.iv_history[symbol].append((datetime.now(), iv))
    
    def get_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """
        Calculate percentile of current IV vs recent history
        
        Low percentile = IV is low = options are cheap = BUY signal
        High percentile = IV is high = options are expensive = SELL signal
        """
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 10:
            return 0.5  # Not enough data, return neutral
        
        history = [iv for _, iv in self.iv_history[symbol]]
        percentile = sum(1 for iv in history if iv < current_iv) / len(history)
        
        return percentile
    
    def get_iv_signal(self, iv_percentile: float) -> float:
        """
        Convert IV percentile to signal score (-1 to +1)
        
        Low IV percentile -> positive signal (buy)
        High IV percentile -> negative signal (sell)
        """
        if iv_percentile < IV_BUY_THRESHOLD:
            # IV is low, options are cheap
            return (IV_BUY_THRESHOLD - iv_percentile) / IV_BUY_THRESHOLD
        elif iv_percentile > IV_SELL_THRESHOLD:
            # IV is high, options are expensive
            return -(iv_percentile - IV_SELL_THRESHOLD) / (1 - IV_SELL_THRESHOLD)
        else:
            # Neutral zone
            return 0.0
    
    def get_mispricing_signal(self, mispricing_pct: float) -> float:
        """
        Convert mispricing percentage to signal score (-1 to +1)
        
        Positive mispricing (underpriced) -> positive signal (buy)
        Negative mispricing (overpriced) -> negative signal (sell)
        """
        # Scale so 10% mispricing = max signal
        signal = np.clip(mispricing_pct * 10, -1, 1)
        return signal
    
    def find_opportunities(
        self,
        quotes: List[OptionQuote],
        min_mispricing: float = None
    ) -> List[Tuple[OptionQuote, PricingResult]]:
        """
        Scan list of quotes for mispricing opportunities
        
        Returns list of (quote, analysis) tuples sorted by mispricing
        """
        if min_mispricing is None:
            min_mispricing = MIN_MISPRICING_PCT
        
        opportunities = []
        
        for quote in quotes:
            try:
                analysis = self.analyze(quote)
                
                if abs(analysis.mispricing_pct) >= min_mispricing:
                    opportunities.append((quote, analysis))
                    
            except Exception as e:
                logger.debug(f"Error analyzing {quote.underlying}: {e}")
        
        # Sort by absolute mispricing (best opportunities first)
        opportunities.sort(key=lambda x: abs(x[1].mispricing_pct), reverse=True)
        
        return opportunities
