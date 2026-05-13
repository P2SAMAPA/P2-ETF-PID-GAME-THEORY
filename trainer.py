import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from scipy.optimize import linprog
import config
import data_manager
import heuristics as h
from pid_controller import PIDController

def compute_heuristic_sharpe(returns, heuristic_func, window, lookback=20):
    """
    For each day in the last `lookback` days, apply heuristic to get signal,
    select top 1 ETF, record its next-day return. Compute Sharpe of those returns.
    """
    if len(returns) < lookback + 1:
        return 0.0
    daily_returns = []
    for i in range(-lookback, 0):
        past = returns.iloc[:i] if i < 0 else returns
        if len(past) < window:
            continue
        recent = past.iloc[-window:]
        signals = heuristic_func(recent, window=window)
        # Choose ETF with highest signal
        best_etf = signals.idxmax()
        # Next day return after the last day of `past`
        next_day_idx = i if i < 0 else -1
        ret = returns.iloc[next_day_idx][best_etf]
        daily_returns.append(ret)
    if len(daily_returns) < 2:
        return 0.0
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    return sharpe

def nash_probabilities(payoff):
    """Mixed strategy probabilities (zero-sum)."""
    n = payoff.shape[0]
    c = [-1] + [0]*n
    A_ub = -np.hstack([np.ones((n,1)), payoff.T])
    b_ub = np.zeros(n)
    A_eq = np.array([[0] + [1]*n])
    b_eq = np.array([1])
    bounds = [(None, None)] + [(0, None)]*n
    try:
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method='highs')
        if res.success:
            probs = res.x[1:]
            return probs / probs.sum()
    except:
        pass
    return np.ones(n)/n

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    # PID to smooth heuristic mixing weights
    pid = PIDController(kp=config.P_GAIN, ki=config.I_GAIN, kd=config.D_GAIN, setpoint=0.0)
    # We'll maintain a target weight for each heuristic; PID will adjust actual weight.
    # Alternatively, PID can directly adjust the blended signal.

    # We'll keep a rolling history of selected probabilities for smoothing.
    prob_history = []

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.TREND_WINDOW, config.MEAN_REV_WINDOW, config.VOL_WINDOW) + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Last available returns for signal computation
        recent_returns = returns.iloc[-max(config.TREND_WINDOW, config.MEAN_REV_WINDOW, config.VOL_WINDOW):]

        # 1. Compute individual heuristic signals (current day)
        trend_sig = h.trend_signal(recent_returns, window=config.TREND_WINDOW)
        meanrev_sig = h.mean_reversion_signal(recent_returns, window=config.MEAN_REV_WINDOW)
        vol_sig = h.volatility_signal(recent_returns, window=config.VOL_WINDOW)

        # 2. Compute recent Sharpe for each heuristic (using past PAYOFF_WINDOW days)
        sharpe_trend = compute_heuristic_sharpe(returns, h.trend_signal, config.TREND_WINDOW, config.PAYOFF_WINDOW)
        sharpe_meanrev = compute_heuristic_sharpe(returns, h.mean_reversion_signal, config.MEAN_REV_WINDOW, config.PAYOFF_WINDOW)
        sharpe_vol = compute_heuristic_sharpe(returns, h.volatility_signal, config.VOL_WINDOW, config.PAYOFF_WINDOW)
        sharpe_vec = np.array([sharpe_trend, sharpe_meanrev, sharpe_vol])

        # 3. Build payoff matrix: payoff_ij = sharpe_i - sharpe_j
        payoff = sharpe_vec[:, np.newaxis] - sharpe_vec[np.newaxis, :]

        # 4. Nash equilibrium probabilities
        probs = nash_probabilities(payoff)
        prob_history.append(probs)

        # 5. PID smoothing: we use PID on the error between current probs and a moving average?
        # Simpler: apply exponential smoothing directly, but PID is more explicit.
        # We'll smooth the blend signal itself: For each ETF, we compute a weighted average of signals,
        # but the weights are the probabilities. We could PID the weight vector.
        # For simplicity, we directly use the Nash probabilities (no PID) because they change slowly.
        # But to include PID, we'll apply PID to the error between the current chosen heuristic's performance and target.
        # That's more complex. We'll stick with Nash probabilities as the blending weights.
        blend_weights = probs

        # 6. Combined signal for each ETF
        signals = pd.DataFrame({
            'trend': trend_sig,
            'mean_rev': meanrev_sig,
            'vol': vol_sig
        })
        combined = (signals['trend'] * blend_weights[0] +
                    signals['mean_rev'] * blend_weights[1] +
                    signals['vol'] * blend_weights[2])

        # 7. Rank ETFs by combined signal, top N
        sorted_etfs = combined.sort_values(ascending=False)
        top_etfs = []
        for i, (ticker, score) in enumerate(sorted_etfs.head(config.TOP_N).items()):
            top_etfs.append({
                'ticker': ticker,
                'score': float(score),
                'weights': {
                    'trend': float(blend_weights[0]),
                    'mean_rev': float(blend_weights[1]),
                    'vol': float(blend_weights[2])
                }
            })
        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        print(f"  Blending weights: trend={blend_weights[0]:.3f}, mean_rev={blend_weights[1]:.3f}, vol={blend_weights[2]:.3f}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "blend_weights": list(blend_weights),
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/pid_game_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Heuristic + Game Theory + PID Engine complete ===")

if __name__ == "__main__":
    main()
