"""

This script will simulate a longer time series with several context switches
for a subset of the agents.

It also calculates the average earned rewards in this series.

Goal: time series with T=200.
T1: coop, t=12
T2: host, t=18
T3: coop, t=24
T4: rand, t=30
T5: coop, t=45
T6: host, t=52
T7: coop, t=64
T8: rand, t=72
T9: host, t=84
T10: coop, t=98
T11: host, t=105
T12: coop, t=124
T13: host, t=134
T14: rand, t=148
T15: coop, t=164
T16: host, t=172
T17: rand, t=184
T18: coop, t=200

author @ale

"""

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from vis import plot_long_mixed_series
import pdb


def run_long_mixed_series(player, plotting='series_only', savefig=False):

    """
    first_context: string, 'friendly', 'hostile', 'random'
    player: string, choose from library
    color_1: string, 'r', 'g', for plotting (make color_1 match first_context)
    color_2: as above, make color_2 match with second context
    plotting: string, 'series_only' will only make time series plot, 'none' will make nothing and 'all' will make all
    """

    # colors - for later:
    coop_col = 'powderblue'
    host_color = 'coral'

    p_shares = {'friendly': 0.8, 'hostile': 0.2, 'neutral': 0.5}
    
    switch = {'T1': 12, 'T2': 18, 'T3': 24, 'T4': 30, 'T5': 45, 'T6': 52, 'T7': 64,
              'T8': 72, 'T9': 84,'T10': 98, 'T11': 105, 'T12': 124 , 'T13': 134}
            #  'T14': 134, 'T15': 164, 'T16': 172, 'T17': 184, 'T18': 200}

    t_switches = list(switch.values())

    context_order = ['friendly', 'hostile', 'friendly', 'neutral', 'friendly', 'hostile', 'friendly',
                     'neutral', 'hostile', 'friendly', 'hostile', 'friendly', 'hostile']
                     #'neutral', 'friendly', 'hostile', 'neutral', 'friendly']

    first_context = context_order[0]
    
    MyEnv = TrustGame(context = first_context )
    
    initial_pshare = MyEnv.p_share
    print('Initial p_share: ',initial_pshare)

    # choose agent model from library
    MyAgent = get_player_agent(player)
    
    MyAgent.beliefs_context = [] 
    MyAgent.partner_behaviour = []

    # ----- RUN INFERENCE LOOP -----
    for t_idx,context in zip(range(0,len(t_switches)-1), context_order):

        MyEnv.p_share = p_shares[context]

        t_start = t_switches[t_idx]
        t_end = t_switches[t_idx+1]

        t = t_end - t_start
        
        run_active_inference_loop(MyAgent, MyEnv, T=t)

    # compute reward and add to MyAgent class. 
    reward_obs = MyAgent.partner_behaviour

    for i in range(len(reward_obs)):
        if reward_obs[i] == 'social':
            reward_obs[i] = 1.5
        elif reward_obs[i] == 'anti-social':
            reward_obs[i] = 0.
        elif reward_obs[i] == 'unknown':
            reward_obs[i] = 0.5

    MyAgent.reward_obs = reward_obs.copy()
    MyAgent.earned_rewards = sum(MyAgent.reward_obs)
    
    # ----- PLOT RESULTS -----
    # depending on where_dots_index: dots appear on green(0), red(1) or grey line(2)

    if plotting == 'series_only':
        plot_long_mixed_series(MyAgent, MyEnv)
        return MyAgent

    if plotting == 'none':
        return MyAgent



if __name__ == '__main__':

    #agents_long_series = ['Player1_healthy', 'Type1_depressed', 'Type2_depressed', 'Type2_social_phobia']

    agents_long_series = ['Type2_social_phobia']
    
    for agent in agents_long_series:

        MyAgent = run_long_mixed_series(player=agent, plotting='series_only')

    


 
