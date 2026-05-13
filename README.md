# Heuristic + Game Theory + PID Switching Engine

Blends three behavioral heuristics (trend, mean‑reversion, volatility) using a zero‑sum game. The Nash equilibrium probabilities determine the blending weights. A PID controller can be applied to smooth weight transitions (optional). Outputs top ETFs per universe.

- **Heuristics:** Trend (20d), Mean‑Reversion (10d), Volatility (21d)
- **Game payoff:** based on recent Sharpe ratio of each heuristic (20-day rolling)
- **Smoothing:** PID (proportional‑integral‑derivative) optional
- **Output:** top 3 ETFs per universe, blending weights

Run daily via GitHub Actions.

## Local execution
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
