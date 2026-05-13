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
    if len(returns) < lookback + 1:
        return 0.0
    daily_returns = []
    for i in range(-lookback, 0):
        past = returns.iloc[:i] if i < 0 else returns
        if len(past) < window:
            continue
        recent = past.iloc[-window:]
        signals = heuristic_func(recent, window=window)
        best_etf = signals.idxmax()
        next_day_idx = i if i < 0 else -1
        ret = returns.iloc[next_day_idx][best_etf]
        daily_returns.append(ret)
    if len(daily_returns) < 2:
        return 0.0
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)
    return sharpe

def nash_probabilities(payoff):
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

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.TREND_WINDOW, config.MEAN_REV_WINDOW, config.VOL_WINDOW) + 50:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        recent_returns = returns.iloc[-max(config.TREND_WINDOW, config.MEAN_REV_WINDOW, config.VOL_WINDOW):]

        trend_sig = h.trend_signal(recent_returns, window=config.TREND_WINDOW)
        meanrev_sig = h.mean_reversion_signal(recent_returns, window=config.MEAN_REV_WINDOW)
        vol_sig = h.volatility_signal(recent_returns, window=config.VOL_WINDOW)

        sharpe_trend = compute_heuristic_sharpe(returns, h.trend_signal, config.TREND_WINDOW, config.PAYOFF_WINDOW)
        sharpe_meanrev = compute_heuristic_sharpe(returns, h.mean_reversion_signal, config.MEAN_REV_WINDOW, config.PAYOFF_WINDOW)
        sharpe_vol = compute_heuristic_sharpe(returns, h.volatility_signal, config.VOL_WINDOW, config.PAYOFF_WINDOW)
        sharpe_vec = np.array([sharpe_trend, sharpe_meanrev, sharpe_vol])

        payoff = sharpe_vec[:, np.newaxis] - sharpe_vec[np.newaxis, :]
        probs = nash_probabilities(payoff)
        blend_weights = probs

        signals = pd.DataFrame({
            'trend': trend_sig,
            'mean_rev': meanrev_sig,
            'vol': vol_sig
        })
        combined = (signals['trend'] * blend_weights[0] +
                    signals['mean_rev'] * blend_weights[1] +
                    signals['vol'] * blend_weights[2])

        # Get all scores dictionary
        all_scores = combined.to_dict()
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
            "all_scores": all_scores,   # added for full ranking
            "blend_weights": list(blend_weights),
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/pid_game_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Heuristic + Game Theory + PID Engine complete ===")

if __name__ == "__main__":
    main()
