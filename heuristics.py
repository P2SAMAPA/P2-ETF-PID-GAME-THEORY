import numpy as np
import pandas as pd

def trend_signal(returns, window=20):
    """
    Trend following: return's momentum (sign of linear regression slope over window).
    """
    signals = pd.Series(index=returns.columns, dtype=float)
    for col in returns.columns:
        y = returns[col].iloc[-window:].values
        if len(y) < window:
            signals[col] = 0.0
        else:
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            signals[col] = np.tanh(slope * 10)  # bounded between -1 and 1
    return signals

def mean_reversion_signal(returns, window=10):
    """
    Mean reversion: negative of z-score of last return relative to recent mean.
    """
    signals = pd.Series(index=returns.columns, dtype=float)
    for col in returns.columns:
        recent = returns[col].iloc[-window:]
        if len(recent) < window:
            signals[col] = 0.0
        else:
            z = (recent.iloc[-1] - recent.mean()) / (recent.std() + 1e-8)
            signals[col] = -np.tanh(z)  # negative because we buy when return is low (reversion)
    return signals

def volatility_signal(returns, window=21):
    """
    Volatility based: low volatility -> high signal (prefer stable), high volatility -> low signal.
    """
    signals = pd.Series(index=returns.columns, dtype=float)
    for col in returns.columns:
        vol = returns[col].iloc[-window:].std()
        if vol == 0:
            signals[col] = 0.0
        else:
            # Normalise across ETFs: lower vol = higher signal
            # We'll compute cross-sectional percentile later; for now, return raw volatility
            signals[col] = -vol   # negative because lower vol gives higher signal? Actually we want positive for low vol.
            # Better: use rank. We'll do scaling inside trainer.
    return signals
