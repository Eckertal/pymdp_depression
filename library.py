# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:13 2022

@author: janik & ale
"""
from pymdp.agent import Agent
from gms import GenerativeModel
import numpy as np

def get_player_agent(name):
    """
    :param name: Name of Player 
    :return: Player instance with generative model depending on player name 
    :rtype: pymdp.agent Agent instance

    """
    
    Player = GenerativeModel(p_share_friendly = 0.9, p_share_hostile = 0.15, p_share_random = 0.5) 

    action_selection = 'deterministic'
    

    if name == 'Player1_healthy':
        """
        This is meant to model a healthy Person.
        """
        #reward_obs_states = [1.0, 0.0, 0.5]  
        #Player.gen_C(p_r0=3.0, p_r1=-2.5, p_r2=1.0) #p_r0=0.9, p_r1= -1.5, p_r2=1.5 works well if lr_pA = 0.4, lr_pB = 0.5 and pr_context_pos=0.55, pr_context_neg=0.45
        Player.gen_C(p_r0=2.5, p_r1=-2.2, p_r2=1.0)
        #context
        Player.gen_D(pr_context_pos=0.6, pr_context_neg=0.35)
        
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name


    elif name == 'biased_A':

        # unclear observation model
        Player.gen_A(p_share_friendly=0.5, p_share_hostile=0.5, p_share_random=0.5)
        Player.gen_B()
        #reward_obs_states = [1.0, 0.0, 0.5]  
        Player.gen_C(p_r0=2.5, p_r1=-2.2, p_r2=1.0)
        #context
        Player.gen_D(pr_context_pos=1/3, pr_context_neg=1/3)
        
        #construct agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)   
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name

        

    elif name == 'biased_B':
        
        Player.gen_depressedB()
        Player.gen_C(p_r0=2.5, p_r1=-2.2, p_r2=1.0) # v.1: p_r0=2.0, p_r1=-1.5, p_r2=1.0
        Player.gen_D(pr_context_pos=1/3, pr_context_neg=1/3)

        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection,  
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=.5, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name


    elif name == 'biased_C':

        # healthy: p_r0=2.5, p_r1=-2.2, p_r2=1.0
        Player.gen_C(p_r0=2.5, p_r1= -5.0, p_r2=1.0)
        Player.gen_D(pr_context_pos=1/3, pr_context_neg=1/3)
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection,  
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=1.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name


    elif name == 'biased_D':
        
        Player.gen_C(p_r0=2.5, p_r1=-2.2, p_r2=1.0) # v1: p_r0=2.0, p_r1=-1.5, p_r2=1.0
        Player.gen_D(pr_context_pos=0.1, pr_context_neg=0.7)
        
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name



    if name == 'Type1_depressed':
        """
        This is meant to model a depressed Person.
        reward-insensitive and pessimistic
        """
        #reward_obs_states = [1.0, 0.0, 0.5]
        Player.gen_depressedB()
        Player.gen_C(p_r0=.8, p_r1=-2.2, p_r2=1.0) #p_r0=0.9, p_r1= -1.5, p_r2=1.5 works well if lr_pA = 0.4, lr_pB = 0.5 and pr_context_pos=0.55, pr_context_neg=0.45
        #context
        Player.gen_D(pr_context_pos=0.15, pr_context_neg=0.8)
        
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = False
        MyAgent.updateD = True
        MyAgent.name = name


    elif name == 'Type2_depressed':
        """
        fatalistic
        """
        Player.gen_staticB()
        Player.gen_C(p_r0=2.5, p_r1=-2.5, p_r2=1.5)
        # (p_r0=3.0, p_r1=-2.5, p_r2=1.0)
        #context
        Player.gen_D(pr_context_pos=0.1, pr_context_neg=0.6)
        
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection,  
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=1.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)

        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name



    if name == 'Type1_social_phobia':
        """
        Social phobia with a focus on C and D uncertainty.

        This is the insecure-avoidant subtype. 
 
        """
        #reward_obs_states = [1.0, 0.0, 0.5]
        # ALE REV: changed p_share_hostile from 0.4 to 0.15!!
        # REV: p_share_random from .5 to .4
        Player.gen_A(p_share_friendly=0.6, p_share_hostile=0.15, p_share_random=0.4) # previously normal A. 
        Player.gen_insecureB() # used to be normal gen_B(), now made transitions into random more likely. 
        Player.gen_C(p_r0=2.5, p_r1=-3.5, p_r2=1.) #p_r0=0.9, p_r1= -1.5, p_r2=1.5 works well if lr_pA = 0.4, lr_pB = 0.5 and pr_context_pos=0.55, pr_context_neg=0.45
        #context
        Player.gen_D(pr_context_pos=0.4, pr_context_neg=0.2)
        
        #constr agent
        # ALE REV: I also made action_selection = stochastic here!!!
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, #alpha=16.0,
                        use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection, #sampling_mode="marginal", 
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=3.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = False
        MyAgent.updateD = True
        MyAgent.name = name
        
           

    elif name == 'Type2_social_phobia':
        """
        Social phobia with a focus loss avoidance and a pessimisitc prior. 

        this is the trauma-and-defeat subtype. 
        
        """
        Player.gen_A(p_share_friendly=0.6, p_share_hostile=0.4, p_share_random=0.5)
        Player.gen_defeatedB()
        Player.gen_C(p_r0=1.2, p_r1= -2.2, p_r2=1.3) # ALE REV less risk averse
        Player.gen_D(pr_context_pos=0.2, pr_context_neg=0.8) # ALE REV higher prior on coop
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection,  
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=1.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name

    elif name == 'Type1_Borderline':
        """
        Borderline Personality Disorder Agent
        """

        Player.gen_A(p_share_friendly=0.6, p_share_hostile=0.4, p_share_random=0.5)
        Player.gen_depressedB()
        Player.gen_C(p_r0=1.0, p_r1= -4.0, p_r2=1.2)
        Player.gen_D(pr_context_pos=0.15, pr_context_neg=0.8)
        #constr agent
        MyAgent = Agent(A=Player.A, B=Player.B, C=Player.C, D=Player.D, E=None, 
                        pA=Player.A, pB=np.array(Player.B,dtype='object'), pD=Player.D, 
                        policy_len=1, inference_horizon=1, 
                        control_fac_idx=None, policies=None, 
                        gamma=16.0, use_utility=True, 
                        use_states_info_gain=True, use_param_info_gain=True, 
                        action_selection=action_selection,  
                        inference_algo="VANILLA", inference_params=None, 
                        modalities_to_learn=[0,1], 
                        lr_pA=0.1   , factors_to_learn="all", lr_pB=1.0, lr_pD=1.0, 
                        use_BMA=True, policy_sep_prior=False, save_belief_hist=True)
        # update gm?
        MyAgent.updateA = True
        MyAgent.updateB = True
        MyAgent.updateD = True
        MyAgent.name = name
              
    
    return MyAgent
