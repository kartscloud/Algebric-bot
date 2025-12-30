"""
================================================================================
GRAPH LAPLACIAN SIGNAL GENERATOR
================================================================================

Implements the graph diffusion approach for detecting mispriced assets.

Theory:
-------
1. Build a weighted graph where:
   - Nodes = underlying assets
   - Edges = correlation between assets
   - Edge weights = |correlation|

2. Compute the Graph Laplacian:
   - L = D - W (unnormalized)
   - L_sym = I - D^(-1/2) W D^(-1/2) (symmetric normalized)
   
3. Apply diffusion operator:
   - h = (I - αL)^J x
   - Where x is the current signal (e.g., recent returns)
   - h is the "smoothed" signal (what the neighborhood expects)
   
4. The residual e = x - h identifies:
   - Positive residual: asset outperforming its correlated peers (potentially overvalued)
   - Negative residual: asset underperforming its correlated peers (potentially undervalued)

For OPTIONS, we interpret this as:
- Stock with positive residual -> PUT signal (stock may revert down)
- Stock with negative residual -> CALL signal (stock may revert up)

================================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import logging

from config import (
    CORRELATION_WINDOW, DIFFUSION_ALPHA, DIFFUSION_STEPS,
    ALL_SYMBOLS
)

logger = logging.getLogger(__name__)


class PriceBuffer:
    """
    Maintains rolling price history for multiple symbols
    """
    
    def __init__(self, max_length: int = 500):
        self.max_length = max_length
        self.prices: Dict[str, deque] = {}
        self.returns: Dict[str, deque] = {}
        
    def update(self, symbol: str, price: float, timestamp: datetime = None):
        """Add new price observation"""
        if timestamp is None:
            timestamp = datetime.now()
            
        if symbol not in self.prices:
            self.prices[symbol] = deque(maxlen=self.max_length)
            self.returns[symbol] = deque(maxlen=self.max_length - 1)
        
        # Calculate return if we have previous price
        if len(self.prices[symbol]) > 0:
            prev_price = self.prices[symbol][-1][1]
            if prev_price > 0:
                ret = (price - prev_price) / prev_price
                self.returns[symbol].append((timestamp, ret))
        
        self.prices[symbol].append((timestamp, price))
    
    def get_prices(self, symbol: str, n: int = None) -> np.ndarray:
        """Get recent prices for a symbol"""
        if symbol not in self.prices:
            return np.array([])
        
        prices = [p for _, p in self.prices[symbol]]
        if n is not None:
            prices = prices[-n:]
        return np.array(prices)
    
    def get_returns(self, symbol: str, n: int = None) -> np.ndarray:
        """Get recent returns for a symbol"""
        if symbol not in self.returns:
            return np.array([])
        
        returns = [r for _, r in self.returns[symbol]]
        if n is not None:
            returns = returns[-n:]
        return np.array(returns)
    
    def get_correlation_matrix(
        self, 
        symbols: List[str], 
        window: int = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate correlation matrix for given symbols
        
        Returns:
        --------
        correlation_matrix : np.ndarray
        valid_symbols : List[str] - symbols with enough data
        """
        if window is None:
            window = CORRELATION_WINDOW
        
        # Get returns for each symbol
        returns_data = {}
        valid_symbols = []
        
        for symbol in symbols:
            rets = self.get_returns(symbol, window)
            if len(rets) >= window * 0.8:  # Need at least 80% of window
                returns_data[symbol] = rets[-window:]
                valid_symbols.append(symbol)
        
        if len(valid_symbols) < 2:
            return np.array([]), []
        
        # Build returns matrix
        n = len(valid_symbols)
        min_len = min(len(returns_data[s]) for s in valid_symbols)
        
        returns_matrix = np.zeros((min_len, n))
        for i, symbol in enumerate(valid_symbols):
            returns_matrix[:, i] = returns_data[symbol][-min_len:]
        
        # Calculate correlation matrix
        # Handle NaN by replacing with 0
        returns_matrix = np.nan_to_num(returns_matrix, nan=0)
        
        corr = np.corrcoef(returns_matrix.T)
        corr = np.nan_to_num(corr, nan=0)
        
        return corr, valid_symbols


class GraphLaplacianSignal:
    """
    Generates trading signals based on graph diffusion
    
    Key insight: 
    If a stock's returns deviate significantly from what its correlated
    peers would predict, it may be mispriced and due for mean reversion.
    """
    
    def __init__(
        self,
        alpha: float = None,
        diffusion_steps: int = None,
        correlation_window: int = None,
        edge_threshold: float = 0.2
    ):
        """
        Parameters:
        -----------
        alpha : float
            Diffusion strength (0 < alpha < 1)
            Higher = more smoothing toward neighborhood consensus
            
        diffusion_steps : int
            Number of diffusion iterations (J in h = (I-αL)^J x)
            Higher = wider neighborhood influence
            
        correlation_window : int
            Number of periods for correlation calculation
            
        edge_threshold : float
            Minimum correlation to form edge (0-1)
        """
        self.alpha = alpha or DIFFUSION_ALPHA
        self.J = diffusion_steps or DIFFUSION_STEPS
        self.corr_window = correlation_window or CORRELATION_WINDOW
        self.edge_threshold = edge_threshold
        
        self.price_buffer = PriceBuffer()
        
    def update_price(self, symbol: str, price: float):
        """Update price for a symbol"""
        self.price_buffer.update(symbol, price)
    
    def _build_adjacency_matrix(self, corr_matrix: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix from correlation matrix
        
        Uses absolute correlation as edge weight (both positive and negative
        correlations indicate relationship)
        """
        W = np.abs(corr_matrix.copy())
        
        # Apply threshold
        W[W < self.edge_threshold] = 0
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(W, 0)
        
        return W
    
    def _compute_laplacian(self, W: np.ndarray) -> np.ndarray:
        """
        Compute symmetric normalized Laplacian
        
        L_sym = I - D^(-1/2) W D^(-1/2)
        
        This normalized form ensures the diffusion operator has
        eigenvalues in [0, 2], making it stable.
        """
        n = W.shape[0]
        
        # Degree matrix (sum of edge weights)
        d = W.sum(axis=1)
        
        # Handle isolated nodes
        d[d == 0] = 1
        
        # D^(-1/2)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        
        # Normalized adjacency
        W_norm = d_inv_sqrt @ W @ d_inv_sqrt
        
        # Laplacian
        L = np.eye(n) - W_norm
        
        return L
    
    def _apply_diffusion(self, L: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Apply diffusion operator: h = (I - αL)^J x
        
        This smooths the signal x over the graph structure.
        After J iterations, h represents what each node's value
        "should be" based on its neighborhood.
        """
        n = L.shape[0]
        diffusion_op = np.eye(n) - self.alpha * L
        
        h = x.copy()
        for _ in range(self.J):
            h = diffusion_op @ h
        
        return h
    
    def compute_signals(self, symbols: List[str] = None) -> Dict[str, float]:
        """
        Compute graph-based signals for all symbols
        
        Returns:
        --------
        Dict[str, float] : Signal score for each symbol (-1 to +1)
            Positive = underperforming peers (potential CALL signal)
            Negative = outperforming peers (potential PUT signal)
        """
        if symbols is None:
            symbols = ALL_SYMBOLS
        
        # Get correlation matrix
        corr_matrix, valid_symbols = self.price_buffer.get_correlation_matrix(
            symbols, self.corr_window
        )
        
        if len(valid_symbols) < 3:
            logger.debug("Not enough symbols for graph signal")
            return {}
        
        # Build adjacency and Laplacian
        W = self._build_adjacency_matrix(corr_matrix)
        L = self._compute_laplacian(W)
        
        # Get current signal (recent cumulative return as the "state")
        x = np.zeros(len(valid_symbols))
        for i, symbol in enumerate(valid_symbols):
            # Use last 5 periods cumulative return as signal
            returns = self.price_buffer.get_returns(symbol, 5)
            if len(returns) > 0:
                x[i] = np.sum(returns)
        
        # Apply diffusion
        h = self._apply_diffusion(L, x)
        
        # Residual: e = x - h
        # Positive residual = outperforming neighborhood expectation
        # Negative residual = underperforming neighborhood expectation
        e = x - h
        
        # Normalize to [-1, 1]
        if np.std(e) > 0:
            e_normalized = e / (2 * np.std(e))
            e_normalized = np.clip(e_normalized, -1, 1)
        else:
            e_normalized = np.zeros_like(e)
        
        # Build result dict
        # INVERT the signal: underperforming = buy calls, outperforming = buy puts
        signals = {}
        for i, symbol in enumerate(valid_symbols):
            signals[symbol] = -e_normalized[i]  # Negative because we want mean reversion
        
        return signals
    
    def get_signal(self, symbol: str) -> float:
        """Get signal for a single symbol"""
        signals = self.compute_signals()
        return signals.get(symbol, 0.0)
    
    def get_network_metrics(self, symbols: List[str] = None) -> Dict:
        """
        Get network topology metrics
        
        Useful for regime detection (highly connected = correlated = risk-on)
        """
        if symbols is None:
            symbols = ALL_SYMBOLS
        
        corr_matrix, valid_symbols = self.price_buffer.get_correlation_matrix(
            symbols, self.corr_window
        )
        
        if len(valid_symbols) < 3:
            return {}
        
        W = self._build_adjacency_matrix(corr_matrix)
        
        # Network metrics
        n = len(valid_symbols)
        num_edges = np.sum(W > 0) / 2  # Undirected
        max_edges = n * (n - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0
        
        # Average correlation
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        avg_corr = np.mean(corr_matrix[mask])
        
        # Spectral properties
        L = self._compute_laplacian(W)
        eigenvalues = np.linalg.eigvalsh(L)
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
        
        return {
            'num_nodes': n,
            'num_edges': int(num_edges),
            'density': density,
            'avg_correlation': avg_corr,
            'spectral_gap': spectral_gap,
            'symbols': valid_symbols
        }


class MomentumSignal:
    """
    Simple momentum signal for underlyings
    
    Positive momentum -> CALL signal
    Negative momentum -> PUT signal
    """
    
    def __init__(self, lookback: int = 15, threshold: float = 0.003):
        """
        Parameters:
        -----------
        lookback : int
            Number of periods for momentum calculation
        threshold : float
            Minimum move to generate signal (e.g., 0.003 = 0.3%)
        """
        self.lookback = lookback
        self.threshold = threshold
        self.price_buffer = PriceBuffer()
    
    def update_price(self, symbol: str, price: float):
        """Update price"""
        self.price_buffer.update(symbol, price)
    
    def get_signal(self, symbol: str) -> float:
        """
        Get momentum signal for a symbol
        
        Returns:
        --------
        float : Signal score (-1 to +1)
            Positive = bullish momentum (CALL)
            Negative = bearish momentum (PUT)
        """
        returns = self.price_buffer.get_returns(symbol, self.lookback)
        
        if len(returns) < self.lookback * 0.5:
            return 0.0
        
        # Cumulative return over lookback
        momentum = np.sum(returns)
        
        # Scale to [-1, 1]
        # 2% move = max signal
        signal = np.clip(momentum / 0.02, -1, 1)
        
        return signal
    
    def get_volatility(self, symbol: str, window: int = 20) -> float:
        """Get realized volatility (annualized)"""
        returns = self.price_buffer.get_returns(symbol, window)
        
        if len(returns) < 5:
            return 0.0
        
        # Annualize (assuming minute data, ~390 min/day, ~252 days/year)
        return np.std(returns) * np.sqrt(390 * 252)
