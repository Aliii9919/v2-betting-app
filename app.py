import numpy as np
from scipy.stats import poisson
import streamlit as st

class FootballPredictionModel:
    def __init__(self, rho=0.05, home_advantage=1.10):
        """
        rho: Dixon-Coles adjustment parameter
        home_advantage: Standard home team multiplier
        """
        self.rho = rho
        self.gamma = home_advantage

    def dixon_coles_adjustment(self, x, y, lambda_h, mu_a):
        """Applies Dixon-Coles adjustment for low-scoring match outcomes."""
        if x == 0 and y == 0:
            return 1 - lambda_h * mu_a * self.rho
        elif x == 0 and y == 1:
            return 1 + mu_a * self.rho
        elif x == 1 and y == 0:
            return 1 + lambda_h * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        return 1.0

    def predict_match_events(self, home_stats, away_stats):
        # Calculate expected Poisson parameters (lambdas)
        lambda_goals_home = home_stats['attack_goals'] * away_stats['defense_goals'] * self.gamma
        mu_goals_away = away_stats['attack_goals'] * home_stats['defense_goals']

        lambda_corners_home = home_stats['attack_corners'] * (away_stats['defense_corners'] / 5.0) * self.gamma
        mu_corners_away = away_stats['attack_corners'] * (home_stats['defense_corners'] / 5.0)

        lambda_sot_home = home_stats['attack_sot'] * (away_stats['defense_sot'] / 4.0) * self.gamma
        mu_sot_away = away_stats['attack_sot'] * (home_stats['defense_sot'] / 4.0)

        # Expected averages
        exp_goals_h = poisson.mean(lambda_goals_home)
        exp_goals_a = poisson.mean(mu_goals_away)

        exp_corners_h = poisson.mean(lambda_corners_home)
        exp_corners_a = poisson.mean(mu_corners_away)

        exp_sot_h = poisson.mean(lambda_sot_home)
        exp_sot_a = poisson.mean(mu_sot_away)

        predictions = {
            "Goals": {
                "Home Goals": round(exp_goals_h, 2),
                "Away Goals": round(exp_goals_a, 2),
                "Total Goals": round(exp_goals_h + exp_goals_a, 2)
            },
            "Corners": {
                "Home Corners": round(exp_corners_h, 2),
                "Away Corners": round(exp_corners_a, 2),
                "Total Corners": round(exp_corners_h + exp_corners_a, 2)
            },
            "Shots on Target": {
                "Home Shots on Target": round(exp_sot_h, 2),
                "Away Shots on Target": round(exp_sot_a, 2),
                "Total Shots on Target": round(exp_sot_h + exp_sot_a, 2)
            }
        }
        return predictions


# --- STREAMLIT USER INTERFACE ---
if __name__ == "__main__":
    st.set_page_config(page_title="V2 Football Prediction Model", layout="centered")
    
    st.title("⚽ V2 Football Prediction Model")
    st.write("Dixon-Coles & Poisson Distribution Match Stats Engine")

    model = FootballPredictionModel()

    netherlands_profile = {
        'attack_goals': 1.35, 'defense_goals': 1.00,
        'attack_corners': 5.0, 'defense_corners': 4.0,
        'attack_sot': 5.0, 'defense_sot': 4.0,
    }

    morocco_profile = {
        'attack_goals': 1.20, 'defense_goals': 1.10,
        'attack_corners': 8.0, 'defense_corners': 3.5,
        'attack_sot': 6.0, 'defense_sot': 3.5,
    }

    result = model.predict_match_events(netherlands_profile, morocco_profile)

    st.subheader("Match Predictions (Mock Netherlands vs Morocco)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚽ Goals")
        for k, v in result["Goals"].items():
            st.metric(label=k, value=v)

    with col2:
        st.markdown("### 🚩 Corners")
        for k, v in result["Corners"].items():
            st.metric(label=k, value=v)

    with col3:
        st.markdown("### 🎯 Shots on Target")
        for k, v in result["Shots on Target"].items():
            st.metric(label=k, value=v)
    
