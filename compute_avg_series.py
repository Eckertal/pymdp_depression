"""
Plot average time series
author @ale

"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from main_simulation_func import run_simulation
#from vis import plot_average_series
import pdb


#all_agents = ['Player1_healthy','biased_A', 'biased_B', 'biased_C', 'biased_D',
#              'Type1_depressed', 'Type2_depressed',
#              'Type1_social_phobia', 'Type2_social_phobia']


all_agents = ['healthiest', 'sickest']

n_simulations = 20

## Main function to run the simulations: 

def compute_series(first_context, agents=all_agents):

    series_coop, series_host = dict(),dict()

    for agent in all_agents:

        agent_series_coop, agent_series_host = [],[]

        for n in range(n_simulations):

            MyAgent = run_simulation(first_context, player=agent, plotting='none')

            posterior_beliefs = MyAgent.beliefs_context

            coop_beliefs = [i[0] for i in posterior_beliefs]
            host_beliefs = [i[1] for i in posterior_beliefs]
            rand_beliefs = [i[2] for i in posterior_beliefs]

            agent_series_coop += [coop_beliefs]
            agent_series_host += [host_beliefs]

            series_coop[MyAgent.name] = agent_series_coop
            series_host[MyAgent.name] = agent_series_host

    df_res = pd.DataFrame.from_dict([series_coop, series_host])
    
    return df_res

if __name__ == '__main__':
    
    """
    # simulate n_sim rounds of friendly first per agent
    df_name_pn = 'avg_series_pn.csv'
    df_pn = compute_series(first_context='friendly', agents=all_agents)
    df_pn.to_csv(df_name_pn)

    # simulate n_sim rounds of hostile first per agent. 
    df_name_np = 'avg_series_np.csv'
    df_np = compute_series(first_context='hostile', agents=all_agents)
    df_np.to_csv(df_name_np)

    """

    

    df_name_pn = 'avg_series_pn.csv'
    df_pn = compute_series(first_context='friendly', agents=all_agents)
    df_pn.to_csv(df_name_pn)

    df_name_np = 'avg_series_np.csv'
    df_np = compute_series(first_context='hostile', agents=all_agents)
    df_np.to_csv(df_name_np)

            
