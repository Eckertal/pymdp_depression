"""
Compute average rewards - long series.

author @ale

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from mixed_long_series import run_long_mixed_series
from vis import plot_earned_rewards

# set up environment and agents.
all_agents = ['Player1_healthy',
              'Type1_depressed', 'Type2_depressed',
              'Type1_social_phobia', 'Type2_social_phobia']

n_simulations = 20

# function to run simulations and return dataframe with all results. 
def compute_earned_rewards(): 

    earned_rewards_results = dict()

    for agent in all_agents:
        agent_rewards = []
        for n in range(n_simulations):
            MyAgent = run_long_mixed_series(player=agent, plotting='none')
            agent_rewards += [MyAgent.earned_rewards]
            earned_rewards_results[MyAgent.name] = agent_rewards

    # transform into df for plotting
    df = pd.DataFrame.from_dict(earned_rewards_results)
    
    return df 

# compute and plot average earned rewards per agent type
df_coop = compute_earned_rewards()
plot_earned_rewards(df_coop, title='Rewards earned per agent type - cooperative context first', filename='rewards_longSeries.svg')
df_coop.to_csv('earned_rewards_coop_long.csv')
