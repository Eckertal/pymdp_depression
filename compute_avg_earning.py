"""
Compute the average outcome (pay-out) per agent type.

@author ale
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pdb

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from main_simulation_func import run_simulation
from vis import plot_earned_rewards

# set up environment and agents.
all_agents = ['Player1_healthy',
              'biased_A', 'biased_B', 'biased_C', 'biased_D',
              'Type1_depressed', 'Type2_depressed',
              'Type1_social_phobia', 'Type2_social_phobia']

# all disorder-specific agents
all_agents = ['Player1_healthy', 'Type1_depressed', 'Type2_depressed',
              'Type1_social_phobia', 'Type2_social_phobia']

# all transdiagnostic agents (supplements only)
all_agents_td = ['Player1_healthy', 'biased_A', 'biased_B', 'biased_C', 'biased_D']

n_simulations = 20

# function to run simulations and return dataframe with all results. 
def compute_earned_rewards(first_context='friendly'): 

    earned_rewards_results = dict()
    win_results, loss_results = dict(), dict()

    for agent in all_agents:
        agent_rewards = []
        wins, loss = [],[]
        for n in range(n_simulations):
            MyAgent = run_simulation(first_context, player=agent, plotting='none')
            agent_rewards += [MyAgent.earned_rewards]
            earned_rewards_results[MyAgent.name] = agent_rewards
            
            wins += [MyAgent.partner_behaviour.count(1.5)]
            loss += [MyAgent.partner_behaviour.count(0.0)]
            win_results[MyAgent.name] = wins
            loss_results[MyAgent.name] = loss

    # transform into df for plotting
    df = pd.DataFrame.from_dict(earned_rewards_results)
    df_wins = pd.DataFrame.from_dict([win_results, loss_results])
    
    return df, df_wins

# compute and plot average earned rewards per agent type
df_coop, df_wins_coop = compute_earned_rewards()
#plot_earned_rewards(df_coop, title='Rewards earned per agent type - cooperative context first', filename='rewards_coop_first.svg')

df_hostile, df_wins_hostile = compute_earned_rewards(first_context='hostile')
#plot_earned_rewards(df_hostile, title='Rewards earned per agent type - hostile context first', filename='rewards_hostile_first.svg')

df_coop.to_csv('earned_rewards_coop_new.csv')
df_hostile.to_csv('earned_rewards_host_new.csv')

df_wins_coop.to_csv('wins_loss_coop.csv')
df_wins_hostile.to_csv('wins_loss_hostile.csv')



