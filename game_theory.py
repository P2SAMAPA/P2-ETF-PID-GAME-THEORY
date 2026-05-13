import numpy as np
import pandas as pd

def compute_payoff_matrix(returns, heuristic_signals, window=20):
    """
    Payoff matrix: for each pair of heuristics, compute Sharpe of the blended signal.
    Simpler: payoff = average return of the heuristic's best ETF over window? 
    We'll use the Sharpe of the top ETF's signal (long only).
    """
    n_heuristics = len(heuristic_signals)
    payoff = np.zeros((n_heuristics, n_heuristics))
    # For each heuristic i (row) and j (col), we simulate trading using i's signal but payoffs based on j's performance? 
    # Zero-sum: player A chooses heuristic i, player B chooses heuristic j. Payoff = Sharpe of i - Sharpe of j.
    # We'll compute recent Sharpe for each heuristic.
    sharpe = []
    for signals in heuristic_signals:
        # For each day in the payoff window, we would need to know what signal produced.
        # Simplify: use average daily return of the top 3 ETFs weighted by signal strength (positive only).
        # Actually simpler: compute the daily return of a portfolio that goes long the ETF with highest signal each day.
        # We'll approximate: for each day, select top 1 ETF by signal, take that ETF's return.
        # Average over window and compute Sharpe.
        # We'll do this inside trainer because we need daily returns.
        pass
    return payoff

def nash_equilibrium(payoff):
    """Return mixed strategy probabilities from payoff matrix (zero-sum)."""
    # Use linear programming to find Nash equilibrium.
    from scipy.optimize import linprog
    n = payoff.shape[0]
    # Maximize v subject to payoff.T * p >= v, sum(p)=1, p>=0
    c = [-1] + [0]*n
    A_ub = -np.hstack([np.ones((n,1)), payoff.T])  # -v + payoff.T * p <= 0
    b_ub = np.zeros(n)
    A_eq = np.array([[0] + [1]*n])
    b_eq = np.array([1])
    bounds = [(None, None)] + [(0, None)]*n
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
    if res.success:
        probs = res.x[1:]
        return probs / probs.sum()
    else:
        return np.ones(n)/n
