"""
================================================================================
HIDDEN MARKOV MODEL REGIME DETECTION
================================================================================

Detects market regimes using a Gaussian Hidden Markov Model.

Theory:
-------
Markets operate in different "regimes" that aren't directly observable:
- Low volatility (calm/bull)
- Normal volatility
- High volatility (crisis/correction)

The HMM models:
- Hidden states: The unobservable regime
- Emissions: Observable features (returns, volatility)
- Transitions: Probability of moving between regimes

We use this to:
1. Adjust position sizing (smaller in high vol regimes)
2. Adjust stop losses (tighter in high vol)
3. Adjust signal thresholds

Since hmmlearn may not be available, we also implement a simple
volatility-threshold based regime detector as fallback.

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
from enum import IntEnum
import logging

from config import (
    Regime, REGIME_LOOKBACK,
    REGIME_POSITION_MULTIPLIER, REGIME_STOP_MULTIPLIER
)

logger = logging.getLogger(__name__)

# Try to import hmmlearn
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.info("hmmlearn not available, using threshold-based regime detection")


class RegimeDetector:
    """
    Detects market regime using HMM or volatility thresholds
    
    Regimes:
    0 - Low volatility (calm market)
    1 - Normal volatility
    2 - High volatility (stress/crisis)
    """
    
    def __init__(
        self,
        n_states: int = 3,
        lookback: int = None,
        use_hmm: bool = True
    ):
        """
        Parameters:
        -----------
        n_states : int
            Number of hidden states (regimes)
        lookback : int
            Number of periods for model fitting
        use_hmm : bool
            Whether to try using HMM (falls back to thresholds if unavailable)
        """
        self.n_states = n_states
        self.lookback = lookback or REGIME_LOOKBACK
        self.use_hmm = use_hmm and HMM_AVAILABLE
        
        # Data storage
        self.returns_history: deque = deque(maxlen=self.lookback * 2)
        self.volatility_history: deque = deque(maxlen=self.lookback * 2)
        
        # HMM model
        self.model = None
        self.is_fitted = False
        
        # State characteristics (learned from data)
        self.state_means: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}
        self.state_volatilities: Dict[int, float] = {0: 0.10, 1: 0.20, 2: 0.40}
        
        # Volatility thresholds for fallback
        self.vol_thresholds = {
            'low': 0.12,   # Below this = low vol regime
            'high': 0.28   # Above this = high vol regime
        }
        
        # State mapping (to ensure state 0 = lowest vol)
        self.state_mapping: Dict[int, int] = {0: 0, 1: 1, 2: 2}
    
    def update(self, ret: float, vol: float = None):
        """
        Update with new observation
        
        Parameters:
        -----------
        ret : float
            Return for the period
        vol : float
            Volatility (if None, computed from returns)
        """
        self.returns_history.append(ret)
        
        if vol is not None:
            self.volatility_history.append(vol)
        elif len(self.returns_history) >= 10:
            # Calculate rolling volatility
            recent_rets = list(self.returns_history)[-20:]
            vol = np.std(recent_rets) * np.sqrt(252 * 390)  # Annualized
            self.volatility_history.append(vol)
    
    def fit(self):
        """
        Fit the regime model to historical data
        """
        if len(self.returns_history) < self.lookback:
            logger.debug("Not enough data to fit regime model")
            return
        
        returns = np.array(list(self.returns_history))
        
        if len(self.volatility_history) >= self.lookback:
            volatility = np.array(list(self.volatility_history))
        else:
            # Calculate rolling volatility
            volatility = np.zeros(len(returns))
            for i in range(10, len(returns)):
                volatility[i] = np.std(returns[max(0, i-20):i]) * np.sqrt(252 * 390)
        
        if self.use_hmm:
            self._fit_hmm(returns, volatility)
        else:
            self._fit_threshold(volatility)
        
        self.is_fitted = True
    
    def _fit_hmm(self, returns: np.ndarray, volatility: np.ndarray):
        """Fit Gaussian HMM"""
        try:
            # Prepare features
            X = np.column_stack([returns[-self.lookback:], 
                                volatility[-self.lookback:]])
            X = np.nan_to_num(X, nan=0)
            
            # Fit model
            self.model = hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type='full',
                n_iter=100,
                random_state=42
            )
            self.model.fit(X)
            
            # Get state characteristics
            states = self.model.predict(X)
            self._compute_state_characteristics(X, states)
            
        except Exception as e:
            logger.warning(f"HMM fitting failed, using thresholds: {e}")
            self.use_hmm = False
            self._fit_threshold(volatility)
    
    def _fit_threshold(self, volatility: np.ndarray):
        """Fit simple threshold-based model"""
        vol = volatility[-self.lookback:]
        vol = vol[vol > 0]  # Remove zeros
        
        if len(vol) < 10:
            return
        
        # Set thresholds at percentiles
        self.vol_thresholds['low'] = np.percentile(vol, 25)
        self.vol_thresholds['high'] = np.percentile(vol, 75)
    
    def _compute_state_characteristics(self, X: np.ndarray, states: np.ndarray):
        """
        Compute mean return and volatility for each state
        Reorder states so that state 0 = lowest volatility
        """
        state_vols = {}
        state_means = {}
        
        for state in range(self.n_states):
            mask = states == state
            if mask.sum() > 0:
                state_vols[state] = X[mask, 1].mean()  # Volatility column
                state_means[state] = X[mask, 0].mean()  # Returns column
            else:
                state_vols[state] = 0
                state_means[state] = 0
        
        # Sort states by volatility (low to high)
        sorted_states = sorted(state_vols.keys(), key=lambda x: state_vols[x])
        self.state_mapping = {old: new for new, old in enumerate(sorted_states)}
        
        # Store reordered characteristics
        for old_state, new_state in self.state_mapping.items():
            self.state_volatilities[new_state] = state_vols[old_state]
            self.state_means[new_state] = state_means[old_state]
    
    def predict(self) -> int:
        """
        Predict current regime
        
        Returns:
        --------
        int : Regime (0=low vol, 1=normal, 2=high vol)
        """
        if not self.is_fitted or len(self.volatility_history) < 5:
            return 1  # Default to normal
        
        if self.use_hmm and self.model is not None:
            return self._predict_hmm()
        else:
            return self._predict_threshold()
    
    def _predict_hmm(self) -> int:
        """Predict using HMM"""
        try:
            returns = list(self.returns_history)[-20:]
            volatility = list(self.volatility_history)[-20:]
            
            X = np.column_stack([returns, volatility])
            X = np.nan_to_num(X, nan=0)
            
            states = self.model.predict(X)
            current_state = states[-1]
            
            # Apply state mapping
            return self.state_mapping.get(current_state, 1)
            
        except Exception as e:
            logger.debug(f"HMM prediction failed: {e}")
            return self._predict_threshold()
    
    def _predict_threshold(self) -> int:
        """Predict using volatility thresholds"""
        if len(self.volatility_history) < 1:
            return 1
        
        current_vol = self.volatility_history[-1]
        
        if current_vol < self.vol_thresholds['low']:
            return 0  # Low vol
        elif current_vol > self.vol_thresholds['high']:
            return 2  # High vol
        else:
            return 1  # Normal
    
    def predict_proba(self) -> np.ndarray:
        """
        Get probability distribution over regimes
        
        Returns:
        --------
        np.ndarray : Probabilities for each regime [P(low), P(normal), P(high)]
        """
        if not self.is_fitted:
            return np.array([0.25, 0.50, 0.25])
        
        if self.use_hmm and self.model is not None:
            try:
                returns = list(self.returns_history)[-20:]
                volatility = list(self.volatility_history)[-20:]
                
                X = np.column_stack([returns, volatility])
                X = np.nan_to_num(X, nan=0)
                
                probs = self.model.predict_proba(X)
                
                # Reorder according to state mapping
                reordered = np.zeros(self.n_states)
                for old_state, new_state in self.state_mapping.items():
                    reordered[new_state] = probs[-1, old_state]
                
                return reordered
                
            except Exception as e:
                logger.debug(f"HMM proba failed: {e}")
        
        # Fallback: point estimate as probability
        regime = self.predict()
        probs = np.array([0.1, 0.1, 0.1])
        probs[regime] = 0.8
        return probs
    
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current regime
        
        Returns:
        --------
        float : Multiplier (< 1 to reduce size in high vol)
        """
        regime = self.predict()
        return REGIME_POSITION_MULTIPLIER.get(regime, 1.0)
    
    def get_stop_multiplier(self) -> float:
        """
        Get stop loss multiplier based on current regime
        
        Returns:
        --------
        float : Multiplier (< 1 for tighter stops in high vol)
        """
        regime = self.predict()
        return REGIME_STOP_MULTIPLIER.get(regime, 1.0)
    
    def get_regime_signal(self) -> float:
        """
        Get regime-based signal score
        
        Low vol regime = bullish signal (options cheap)
        High vol regime = bearish signal (options expensive)
        
        Returns:
        --------
        float : Signal (-1 to +1)
        """
        probs = self.predict_proba()
        
        # Weight: low vol = +1, normal = 0, high vol = -1
        weights = np.array([1.0, 0.0, -1.0])
        
        return np.dot(probs, weights)
    
    def get_status(self) -> Dict:
        """Get current regime status"""
        regime = self.predict()
        probs = self.predict_proba()
        
        regime_names = {0: 'LOW_VOL', 1: 'NORMAL', 2: 'HIGH_VOL'}
        
        return {
            'regime': regime,
            'regime_name': regime_names.get(regime, 'UNKNOWN'),
            'probabilities': {
                'low_vol': probs[0],
                'normal': probs[1],
                'high_vol': probs[2]
            },
            'position_multiplier': self.get_position_multiplier(),
            'stop_multiplier': self.get_stop_multiplier(),
            'volatility_thresholds': self.vol_thresholds,
            'is_fitted': self.is_fitted,
            'using_hmm': self.use_hmm and self.model is not None
        }


class RegimeAwareRiskManager:
    """
    Adjusts risk parameters based on detected regime
    """
    
    def __init__(self, regime_detector: RegimeDetector):
        self.regime = regime_detector
    
    def adjust_position_size(self, base_size: float) -> float:
        """Adjust position size for regime"""
        return base_size * self.regime.get_position_multiplier()
    
    def adjust_stop_loss(self, base_stop: float) -> float:
        """Adjust stop loss for regime (tighter in high vol)"""
        return base_stop * self.regime.get_stop_multiplier()
    
    def should_reduce_exposure(self) -> bool:
        """Check if we should reduce overall exposure"""
        probs = self.regime.predict_proba()
        
        # Reduce if high vol probability > 60%
        if probs[2] > 0.6:
            return True
        
        # Also reduce if transitioning to high vol
        if probs[2] > 0.4 and probs[1] > 0.3:
            return True
        
        return False
    
    def get_max_positions(self, base_max: int) -> int:
        """Get maximum positions for current regime"""
        regime = self.regime.predict()
        
        if regime == 2:  # High vol
            return max(1, base_max // 2)
        elif regime == 0:  # Low vol
            return base_max + 1
        else:
            return base_max
