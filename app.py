import numpy as np
from scipy.stats import poisson

class FootballPredictionModel:
    def __init__(self, rho=0.05, home_advantage=1.2):
        """
        rho: Dixon-Coles adjustment factor for low scorelines.
        home_advantage: Standard home ground multiplier (e.g., 1.2x baseline).
        """
        self.rho = rho
        self.gamma = home_advantage

    def dixon_coles_adjustment(self, x, y, lambda_h, mu_a):
        """Applies Dixon-Coles adjustment tau(x,y) to fix independence flaw."""
        if x == 0 and y == 0:
            return 1 - lambda_h * mu_a * self.rho
        elif x == 0 and y == 1:
            return 1 + mu_a * self.rho
        elif x == 1 and y == 0:
            return 1 + lambda_h * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        else:
            return 1.0

    def predict_match_events(self, team_h_stats, team_a_stats, max_events=8):
        """
        Accepts data dictionaries for both teams containing estimated event rates.
        """
        # 1. Calculate Expected Goals (lambda for Home, mu for Away)
        lambda_goals = team_h_stats['attack_goals'] * team_a_stats['defense_goals'] * self.gamma
        mu_goals = team_a_stats['attack_goals'] * team_h_stats['defense_goals']
        
        # 2. Build Scoreline Probability Matrix (Dixon-Coles)
        score_matrix = np.zeros((max_events, max_events))
        for x in range(max_events):
            for y in range(max_events):
                p_x = poisson.pmf(x, lambda_goals)
                p_y = poisson.pmf(y, mu_goals)
                tau = self.dixon_coles_adjustment(x, y, lambda_goals, mu_goals)
                score_matrix[x, y] = p_x * p_y * tau
                
        # Normalize matrix to ensure probabilities sum to 1
        score_matrix /= np.sum(score_matrix)

        # 3. Calculate Match Outcome Probabilities (1 / X / 2)
        prob_home_win = np.sum(np.tril(score_matrix, -1))
        prob_draw = np.sum(np.diag(score_matrix))
        prob_away_win = np.sum(np.triu(score_matrix, 1))

        # 4. Get Top 3 Correct Scores
        top_scores = []
        # Flatten the matrix and get the indices of the top 3 highest probabilities
        flat_indices = np.argsort(score_matrix.ravel())[::-1][:3]
        for idx in flat_indices:
            x, y = np.unravel_index(idx, score_matrix.shape)
            prob = score_matrix[x, y] * 100
            top_scores.append((f"{x}-{y}", f"{prob:.2f}%"))

        # 5. Model Secondary Events: Corners and Shots on Target
        lambda_corners = team_h_stats['attack_corners'] * team_a_stats['defense_corners'] * 1.1 
        mu_corners = team_a_stats['attack_corners'] * team_h_stats['defense_corners']
        
        lambda_sot = team_h_stats['attack_sot'] * team_a_stats['defense_sot'] * 1.15
        mu_sot = team_a_stats['attack_sot'] * team_h_stats['defense_sot']

        predictions = {
            "Outcomes (1/X/2)": {
                "Home Win (1)": f"{prob_home_win * 100:.2f}%",
                "Draw (X)": f"{prob_draw * 100:.2f}%",
                "Away Win (2)": f"{prob_away_win * 100:.2f}%"
            },
            "Top 3 Correct Scores": {
                f"Rank {i+1}": f"{score} ({pct})" for i, (score, pct) in enumerate(top_scores)
            },
            "Expected Totals": {
                "Home Goals": round(lambda_goals, 2),
                "Away Goals": round(mu_goals, 2),
                "Home Corners": round(lambda_corners, 1),
                "Away Corners": round(mu_corners, 1),
                "Home Shots on Target": round(lambda_sot, 1),
                "Away Shots on Target": round(mu_sot, 1)
            }
        }
        return predictions

# --- RUNNING THE MOCK NETHERLANDS VS MOROCCO TEST ---
if __name__ == "__main__":
    model = FootballPredictionModel()

    netherlands_profile = {
        'attack_goals': 1.35,   'defense_goals': 0.95, 
        'attack_corners': 5.0,  'defense_corners': 8.0,
        'attack_sot': 5.0,      'defense_sot': 4.0
    }

    morocco_profile = {
        'attack_goals': 1.20,   'defense_goals': 1.05, 
        'attack_corners': 8.0,  'defense_corners': 5.0,
        'attack_sot': 6.0,      'defense_sot': 3.5
    }

    result = model.predict_match_events(netherlands_profile, morocco_profile)
    
    for category, stats in result.items():
        print(f"\n🔹 {category}:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
      
