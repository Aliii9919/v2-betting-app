import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson

# Page Configuration
st.set_page_config(page_title="V2 Football Prediction Model", layout="wide")

st.title("⚽ V2 Football Prediction Model")
st.write("Poisson Distribution & Expected Goals Engine")

# ==========================================
# 1. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Match Settings")

home_team = st.sidebar.text_input("Home Team", "Netherlands")
away_team = st.sidebar.text_input("Away Team", "Morocco")

st.sidebar.markdown("---")
st.sidebar.header("Team Strengths")

# League average goals per team per match
league_avg = st.sidebar.number_input("League Avg Goals / Match", min_value=0.5, max_value=4.0, value=1.35, step=0.05)

col_h1, col_h2 = st.sidebar.columns(2)
with col_h1:
    home_attack = st.sidebar.number_input(f"{home_team} Attack Rating", value=1.20, step=0.05)
with col_h2:
    home_defense = st.sidebar.number_input(f"{home_team} Defense Rating", value=0.85, step=0.05)

col_a1, col_a2 = st.sidebar.columns(2)
with col_a1:
    away_attack = st.sidebar.number_input(f"{away_team} Attack Rating", value=1.10, step=0.05)
with col_a2:
    away_defense = st.sidebar.number_input(f"{away_team} Defense Rating", value=0.95, step=0.05)


# ==========================================
# 2. EXPECTED GOALS (xG) ENGINE
# ==========================================
home_xg = home_attack * away_defense * league_avg
away_xg = away_attack * home_defense * league_avg

st.subheader(f"📊 Expected Goals (xG): {home_team} vs {away_team}")
col1, col2 = st.columns(2)
col1.metric(f"{home_team} xG", f"{home_xg:.2f}")
col2.metric(f"{away_team} xG", f"{away_xg:.2f}")

st.markdown("---")


# ==========================================
# 3. POISSON MATRIX & PROBABILITIES
# ==========================================
def calculate_predictions(home_xg, away_xg, max_goals=6):
    # Calculate goal probability distribution up to max_goals
    home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
    away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
    
    # Outer product to build scoreline grid
    score_matrix = np.outer(home_probs, away_probs)
    
    # 1X2 Probabilities
    home_win_prob = np.sum(np.tril(score_matrix, -1))
    draw_prob = np.sum(np.diag(score_matrix))
    away_win_prob = np.sum(np.triu(score_matrix, 1))
    
    # Over / Under 2.5 Goals
    total_goals_grid = np.fromfunction(lambda i, j: i + j, (max_goals, max_goals), dtype=int)
    over_25_prob = np.sum(score_matrix[total_goals_grid > 2.5])
    under_25_prob = 1.0 - over_25_prob
    
    # Both Teams to Score (BTTS)
    btts_yes_prob = np.sum(score_matrix[1:, 1:])
    btts_no_prob = 1.0 - btts_yes_prob
    
    # Exact Scores List
    scores_list = []
    for h in range(max_goals):
        for a in range(max_goals):
            prob = score_matrix[h, a]
            fair_odds = 1 / prob if prob > 0 else 0
            scores_list.append({
                "Score": f"{h} - {a}",
                "Probability": f"{prob * 100:.1f}%",
                "Fair Odds": f"{fair_odds:.2f}",
                "_raw_prob": prob
            })
    
    # Sort exact scores by highest probability
    scores_df = pd.DataFrame(scores_list).sort_values(by="_raw_prob", ascending=False).drop(columns=["_raw_prob"])
    
    return {
        "home_win": home_win_prob,
        "draw": draw_prob,
        "away_win": away_win_prob,
        "over_25": over_25_prob,
        "under_25": under_25_prob,
        "btts_yes": btts_yes_prob,
        "btts_no": btts_no_prob,
        "scores_df": scores_df
    }

results = calculate_predictions(home_xg, away_xg)

# ==========================================
# 4. DISPLAY RESULTS IN DASHBOARD
# ==========================================

# 1X2 Market Display
st.subheader("🎯 1X2 Match Outcome Probabilities")
m1, m2, m3 = st.columns(3)

m1.metric(
    f"{home_team} Win", 
    f"{results['home_win']*100:.1f}%", 
    f"Odds: {1/results['home_win']:.2f}"
)
m2.metric(
    "Draw", 
    f"{results['draw']*100:.1f}%", 
    f"Odds: {1/results['draw']:.2f}"
)
m3.metric(
    f"{away_team} Win", 
    f"{results['away_win']*100:.1f}%", 
    f"Odds: {1/results['away_win']:.2f}"
)

st.markdown("---")

# Secondary Markets (Over/Under & BTTS)
c1, c2 = st.columns(2)

with c1:
    st.subheader("⚽ Over / Under 2.5 Goals")
    st.write(f"**Over 2.5:** {results['over_25']*100:.1f}% (Fair Odds: {1/results['over_25']:.2f})")
    st.write(f"**Under 2.5:** {results['under_25']*100:.1f}% (Fair Odds: {1/results['under_25']:.2f})")

with c2:
    st.subheader("🥅 Both Teams To Score (BTTS)")
    st.write(f"**Yes:** {results['btts_yes']*100:.1f}% (Fair Odds: {1/results['btts_yes']:.2f})")
    st.write(f"**No:** {results['btts_no']*100:.1f}% (Fair Odds: {1/results['btts_no']:.2f})")

st.markdown("---")

# Top Correct Scores Table
st.subheader("🏆 Top Correct Score Predictions")
st.dataframe(results["scores_df"].head(8), hide_index=True, use_container_width=True)
