import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="PID Game Theory Engine", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 1rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Heuristic + Game Theory + PID Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Trend, Mean‑Reversion, Volatility | Zero‑sum game selects heuristics | PID smooths switching</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧠 Game Theory PID")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Heuristics:** Trend (20d), Mean‑Reversion (10d), Volatility (21d)")
st.sidebar.markdown("**Game payoff:** Recent Sharpe difference")
st.sidebar.markdown("**PID:** smooth blending weights")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'pid_game_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Blended Heuristic Signal")
st.markdown("*Signals combined using Nash equilibrium probabilities from a zero‑sum game among heuristics.*")

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">score = {etf['score']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
    # Show blending weights (same for all ETFs in universe)
    if 'blend_weights' in uni_data:
        w = uni_data['blend_weights']
        st.caption(f"Blending weights: Trend={w[0]:.2f}, Mean-Reversion={w[1]:.2f}, Volatility={w[2]:.2f}")

    # --- New: Dropdown table for all ETFs ---
    all_scores = uni_data.get("all_scores", {})
    if all_scores:
        with st.expander("📊 Full ranking (all ETFs)"):
            df_all = pd.DataFrame(list(all_scores.items()), columns=["ETF", "Score"])
            df_all = df_all.sort_values("Score", ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Each day, a zero‑sum game compares recent Sharpe of the three heuristics; the Nash equilibrium gives probabilities used to blend their signals. PID controller can optionally smooth weight changes.")
