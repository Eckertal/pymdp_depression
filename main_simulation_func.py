"""

MAIN function to perform simulations
This script will create the canonical simulations in the paper
with 40 timesteps and one context switch after t=20 timesteps. 
author @ale

"""

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from vis import *


def run_simulation(first_context, player, plotting='series_only', savefig=False):

    """
    first_context: string, 'friendly', 'hostile', 'random'
    player: string, choose from library
    color_1: string, 'r', 'g', for plotting (make color_1 match first_context)
    color_2: as above, make color_2 match with second context
    plotting: string, 'series_only' will only make time series plot, 'none' will make nothing and 'all' will make all
    """

    MyEnv = TrustGame(context = first_context)

    if first_context == 'friendly':
        color_1 = 'powderblue'
        color_2 = 'coral'
        dots_index = 0

    elif first_context == 'hostile':
        color_1 = 'coral'
        color_2 = 'powderblue'
        dots_index = 1
    
    initial_pshare = MyEnv.p_share
    print('Initial p_share: ',initial_pshare)

    # choose a Player model from library e.g. 'Player1'
    MyAgent = get_player_agent(player)
    
    MyAgent.beliefs_context = [] 
    MyAgent.partner_behaviour = []
    #choose time intervall for the simulation
    T1=20

    # ----- RUN INFERENCE LOOP -----
    # get prior settings of D
    A_pre = MyAgent.A[0]
    B_pre = MyAgent.B[0]
    D_pre = MyAgent.D[0]
    run_active_inference_loop(MyAgent, MyEnv, T = T1)
    D_post_first = MyAgent.D[0]

    # context switch
    MyEnv.p_share = 1 - initial_pshare
    T2 =20
    run_active_inference_loop(MyAgent, MyEnv, T = T2)

    A_post = MyAgent.A[0]
    B_post = MyAgent.B[0]
    D_post = MyAgent.D[0]

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
    if plotting == 'none':
        return MyAgent
    
    if plotting == 'all': 
        plot_results_context_change(T1, T2, color_1, color_2, MyAgent, MyEnv, where_dots_idx=dots_index)
        if savefig:
            plt.savefig('timeseries.png')
        plot_A_pre_post(A_pre, A_post)
        plot_B_pre_post(B_pre, B_post)
        plot_D_pre_post(D_pre, D_post)
        return MyAgent

    elif plotting == 'series_only':
        #plot_results_context_change(T1, T2, color_1, color_2, MyAgent, MyEnv, where_dots_idx=1)
        fancy_time_series(T1, T2, color_1, color_2, MyAgent, MyEnv, where_dots_idx=dots_index)
        return MyAgent


if __name__ == '__main__':

    player_list = ['Type2_social_phobia']

    for player in player_list: 

        MyAgent = run_simulation(first_context='friendly', player=player, plotting='series_only')
        MyAgent = run_simulation(first_context='hostile', player=player, plotting='series_only')

    


 
