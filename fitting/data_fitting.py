# fitting the model.

import sys, pdb, glob, os, pickle
import argparse
import multiprocessing
root = os.path.dirname(os.getcwd())
sys.path.append(root)

from library import GenerativeModel
from inference import run_inference_opt
from inference import run_active_inference_loop
from scipy.special import softmax
import scipy.optimize as opt

from pymdp.agent import Agent

import pandas as pd
import numpy as np


root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)


"""
author @ALE. 
1. make function that inits model with parameters
2. define cost function
3. optimize on data. 
"""

"""
parameters.

A: Player.gen_A(p_share_friendly, p_share_hostile, p_share_random)
B: Player.gen_B(self) --> p_ff, p_fh, p_hf, p_hh, p_rf, p_rh
C: Player.gen_C(p_r0, p_r1, p_r2)
D: Player.gen_D(pr_context_pos, pr_context_neg)
"""

parser = argparse.ArgumentParser(
                    prog='data_fitting',
                    description='fitting data',
                    epilog='Text at the bottom of help')

parser.add_argument('-n', '--numprocs', action='store', default=1)
args = parser.parse_args()

num_calls = 0

def cost_function(p_sim_share, r_t):
    # minimize cost between simulated actions and real responses
    # p_sim_share sollte prob of share action sein
    # in the r_t, 0 means share
    clip_val = 0.001
    r_t = (1.0-np.array(r_t)).clip(clip_val,1.0-clip_val)
    p_sim_share = np.array(p_sim_share).clip(clip_val, 1.0-clip_val)

    max_cost_per_trial = clip_val*np.log(clip_val/(1.0-clip_val)) + (1.0-clip_val)*np.log((1.0-clip_val)/clip_val)
    
    cost = r_t * np.log(r_t/p_sim_share) + (1.0-r_t) * np.log((1.0-r_t) / (1.0-p_sim_share))
 
    cost=cost.mean()
    
    return cost/max_cost_per_trial

def objective(params, data):

    # unpack 14 parameters.
    (logit_share_f, logit_share_h, logit_share_r,
    p_ff, p_fh, p_hf, p_hh, p_rf, p_rh,
    #p_r0, p_r1, p_r2,
    pr_context_pos, pr_context_neg) = params

    # init agent and gms
    Player = GenerativeModel(p_share_friendly=0.9, p_share_hostile=0.15, p_share_random=0.5)
    action_selection='deterministic'

    Player.gen_A_opt(logit_share_f, logit_share_h, logit_share_r)
    Player.gen_B_opt(p_ff, p_fh, p_hf, p_hh, p_rf, p_rh)
    #Player.gen_C(p_r0, p_r1, p_r2)
    Player.gen_C(2.0,-2.0,0.0)
    Player.gen_D(pr_context_pos, pr_context_neg)

    MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=0.1, alpha=5.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)

    # read data, add start step
    reward      = [2]+list(data['reward'])
    partner_obs = [2]+list(data['partnerAnswer'])
    responses   = [2]+list(data['response'])  

    # set up cost - penalty for num of params
    cost = (np.array(params) ** 2).mean() * 0.01

    # run inference with our Agent and current obs
    all_observations = np.array([reward, partner_obs, responses]).T
    
    sim_actions = run_inference_opt(MyAgent, all_observations)

    # minimize cost between observed responses (patients) and simulated action probabilities
    cost += cost_function(sim_actions[:-1], responses[1:])
    print("cost", cost)
    sys.stdout.flush()

    return cost

def opt_worker(args):

    sbj, sbj_try, x0, bounds, data = args

    def d_obj(params):
        return objective(params, data)

    r = opt.minimize(
        d_obj,
        x0,
        method='Powell',
        bounds=bounds,
        tol=1e-3,
        options={"maxiter":20, "disp":True},
        )

    if "args" in r:
        del r.specs["args"]["func"]

    print(f"Subject {sbj}, try {sbj_try} done")
    
    print(objective(r.x,data))

    r["sbj"] = sbj
    r["sbj_try"] = sbj_try

    return r


if __name__ == "__main__":

    path_to_data = os.path.join(root, "fitting")
    path_to_results = os.path.join(root, "fitting", "opt_results")

    os.chdir(path_to_data)

    df = pd.read_csv(os.path.join(path_to_data, "data_inclBDI_n25.csv"))

    subjects = list(df['Participant Private ID'].unique())

    tries_per_subject = 10
    worker_args = []
    
    for sbj in subjects:
        num_calls = 0

        df_sbj = df[df['Participant Private ID'] == sbj]

        reward = list(df_sbj['reward'])
        partnerAnswer = list(df_sbj['partnerAnswer'])
        response = list(df_sbj['Response'])

        # extreme data: always share, always lose
        reward = [0]*len(reward)
        partnerAnswer = [0]*len(partnerAnswer)
        response = [0]*len(response)

        data = {
            "reward": reward, 
            "partnerAnswer": partnerAnswer,
            "response": response
            }

        space = [
            (-4.0, 4.0), # friendly share
            (-4.0, 4.0), # hostile share
            (-4.0, 4.0), # random share
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0),
            #(-2.0, 2.0),
            #(-2.0, 2.0),
            #(-2.0, 2.0),
            (-2.0, 2.0),
            (-2.0, 2.0)
            ]

        for i in range(tries_per_subject):
            x0 = [
                (xmin + xmax) / 2.0 + np.random.random() * (xmax - xmin) / 4
                for xmin, xmax in space
            ]
            worker_args.append((sbj, i, x0, space, data))

    # start with the optimizer
    os.chdir(path_to_results)

    num_procs = int(args.numprocs)

    if num_procs == 1:

        for opt_res in map(opt_worker, worker_args):

            sbj = opt_res["sbj"]
            sbj_try = opt_res["sbj_try"]
            filename = str(f"result_{sbj}_{sbj_try}.pkl")
            with open(filename, "wb") as handle:
                pickle.dump(opt_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"########## {sbj}, try {sbj_try} saved ##########")

    else: 

        with multiprocessing.Pool(processes=num_procs) as pool:
            for opt_res in pool.imap_unordered(opt_worker, worker_args):
                sbj = opt_res["sbj"]
                sbj_try = opt_res["sbj_try"]
                filename = str(f"result_{sbj}_{sbj_try}.pkl")
                with open(filename, "wb") as handle:
                    pickle.dump(opt_res, handle, protocol=pickle.HIGHEST_PROTOCOL)

                print(f"########## {sbj}, try {sbj_try} saved ##########")


    
