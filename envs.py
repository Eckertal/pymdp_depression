# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:55:15 2022

@author: janik
"""

import numpy as np
from pymdp import utils

class TrustGame(object):
    """ """
    def __init__(self, context = None, neutral_hstate = True):
        """

        :param context:  (Default value = None)

        """
        self.neutral_hstate = neutral_hstate
        
        #init p_share depending on context
        self._init_p_share(context)        
        #hidden states
        if self.neutral_hstate is True:
            self.context_states = ['friendly','hostile', 'neutral']
        else:
            self.context_states = ['friendly','hostile']
            
        self.choice_states = ['share', 'keep', 'start']
        # n hidden states` and num_hidden_factors
        self.num_hstates = [len(self.context_states), len(self.choice_states)]
        self.num_hfactors = len(self.num_hstates)
        
        #action names
        self.choice_action_names = ['share', 'keep', 'start']
        self.context_action_names = ['do-nothing']
               
        #observation states    
        self.reward_obs_states = [1.0, 0.0, 0.5]   
        self.behaviour_obs_states = ['social', 'anti-social', 'unknown']
        self.choice_obs_states = ['share', 'keep', 'start']
        
        # init dimensions 
        """ Define `num_obs` and `num_modalities` below """
        self.num_obs = [len(self.reward_obs_states), len(self.behaviour_obs_states), len(self.choice_obs_states)]
        self.num_modalities = len(self.num_obs)
        
        #store
        self.partner_behaviour = []
        
        
    def _init_p_share(self,context):
        """

        :param context: 

        """
        if context is None:
            self.context = ['friendly', 'hostile', 'random'][np.random.randint(0,3)]
        else:
            self.context = context
        if self.context == 'friendly':
            self.p_share = 0.8            
        elif self.context == 'hostile':
            self.p_share = 0.2
        elif self.context == 'random':
            self.p_share = 0.5
        else:
            raise ValueError(f'"{self.context}" is not a configured context.')
        
    def step(self,action):
        """

        :param action: 

        """
        if action == 'start':
            choice_obs = action
            reward_obs = self.reward_obs_states[2]
            behaviour_obs = self.behaviour_obs_states[2]
            
        elif action == 'keep':
            choice_obs = action
            reward_obs = self.reward_obs_states[2]
            behaviour_obs = self.behaviour_obs_states[2]
            
        elif action == 'share':
            choice_obs = action
            #sample partner decision
            if self.context == 'manual':
                answer = input('Press 1 to share (friendly), 0 to keep (hostile).')
                if answer == 'q':
                    return None
                partner_action = int(answer)+1
            else:
                partner_action = utils.sample(np.array([self.p_share, 1-self.p_share, 0.0])) # should be normalized to 1 sample doesnt check for it
            reward_obs = self.reward_obs_states[partner_action]
            behaviour_obs = self.behaviour_obs_states[partner_action]
        
        self.partner_behaviour.append(behaviour_obs)
        obs = [reward_obs, behaviour_obs, choice_obs]
        
        return obs
    
class TrustGameCoop(TrustGame):
    """ """
    def __init__(self, neutral_hstate = True):
        #inherit state and choice names etc. from Trustgame Class
        TrustGame.__init__(self)
        self.player1_count = [0.0]
        self.player2_count = [0.0]
        
    def step(self,actions):
        """
        :param action: 

        """
        action1, action2 = actions
        if action1 == 'start' and action2 == 'start':
            choice_obs1 = action1
            reward_obs1 = self.reward_obs_states[2]
            behaviour_obs1 = self.behaviour_obs_states[2]
            
            choice_obs2 = action2
            reward_obs2 = self.reward_obs_states[2]
            behaviour_obs2 = self.behaviour_obs_states[2]
            
        elif action1 == 'keep' and action2 == 'keep':
            choice_obs1 = action1
            reward_obs1 = self.reward_obs_states[2]
            behaviour_obs1 = self.behaviour_obs_states[1]
            
            choice_obs2 = action2
            reward_obs2 = self.reward_obs_states[2]
            behaviour_obs2 = self.behaviour_obs_states[1]
            
        elif action1 == 'keep' and action2 == 'share':
            choice_obs1 = action1
            reward_obs1 = self.reward_obs_states[0] # 1€ reward
            behaviour_obs1 = self.behaviour_obs_states[0] #observed social behaviour
            
            choice_obs2 = action2
            reward_obs2 = self.reward_obs_states[1] # 0€ reward
            behaviour_obs2 = self.behaviour_obs_states[1] #observed anti-social behaviour   
            
        elif action1 == 'share' and action2 == 'keep':
            choice_obs1 = action1
            reward_obs1 = self.reward_obs_states[1] # 0€ reward
            behaviour_obs1 = self.behaviour_obs_states[1] #observed anti-social behaviour   
            
            choice_obs2 = action2
            reward_obs2 = self.reward_obs_states[0] # 1€ reward
            behaviour_obs2 = self.behaviour_obs_states[0] #observed social behaviour  
            
        elif action1 == 'share' and action2 == 'share':
            choice_obs1 = action1
            reward_obs1 = self.reward_obs_states[0]
            behaviour_obs1 = self.behaviour_obs_states[0]
            
            choice_obs2 = action2
            reward_obs2 = self.reward_obs_states[0]
            behaviour_obs2 = self.behaviour_obs_states[0]            
        
        obs1 = [reward_obs1, behaviour_obs1, choice_obs1]
        obs2 = [reward_obs2, behaviour_obs2, choice_obs2]
        self.player1_count.append(self.player1_count[-1]+reward_obs1)
        self.player2_count.append(self.player2_count[-1]+reward_obs2)
        
        return [obs1, obs2]
    
    
class TrustGameCoopAsymmetric(TrustGame):
    
    def __init__(self, neutral_hstate = True):
        #inherit state and choice names etc. from Trustgame Class
        TrustGame.__init__(self)
        
        #observation states    
        self.reward_obs_states = ['high', 'nothing', 'low'] 
        
        print(self.reward_obs_states)
        #action names
        self.choice_states = ['share', 'keep', 'start', 'observe']
        self.choice_action_names = ['share', 'keep', 'start', 'observe']  
        self.choice_obs_states = ['share', 'keep', 'start', 'observe']
        # init dimensions 
        """ Define `num_obs` and `num_modalities` below """
        self.num_obs = [len(self.reward_obs_states), len(self.behaviour_obs_states), len(self.choice_obs_states)]
        self.num_modalities = len(self.num_obs)
        
        self.player1_count = [0.0]
        self.player2_count = [0.0]
        
    def player1_action(self, action):
        self.player1_choice = action
        if action == 'share':
            player1_trusts = True
        else: 
            player1_trusts = False
        return player1_trusts
        
    def player2_observation(self):
        if self.player1_choice == 'keep':
            choice_obs = 'observe'
            reward_obs = self.reward_obs_states[1]
            behaviour_obs = self.behaviour_obs_states[1]
        
        elif self.player1_choice == 'share':
            choice_obs = 'observe'
            reward_obs = self.reward_obs_states[1] #'unknown'
            behaviour_obs = self.behaviour_obs_states[0]
            
        return [reward_obs, behaviour_obs, choice_obs]
        
    def step(self, action_player2):
        
        if self.player1_choice == 'start':
            choice_obs1 = 'start'
            reward_obs1 = self.reward_obs_states[1]
            behaviour_obs1 = self.behaviour_obs_states[2]  
            
            choice_obs2 = 'start'
            reward_obs2 = self.reward_obs_states[1]
            behaviour_obs2 = self.behaviour_obs_states[2] 
            
        elif self.player1_choice == 'keep':
            choice_obs1 = 'keep'
            reward_obs1 = self.reward_obs_states[2]
            behaviour_obs1 = self.behaviour_obs_states[2]  
            
            choice_obs2 = 'observe'
            reward_obs2 = self.reward_obs_states[1]
            behaviour_obs2 = self.behaviour_obs_states[1]   
            
        elif self.player1_choice == 'share':

            if action_player2 == 'share':
                choice_obs1 = 'share'
                reward_obs1 = self.reward_obs_states[0]
                behaviour_obs1 = self.behaviour_obs_states[0]  
                
                choice_obs2 = 'share'
                reward_obs2 = self.reward_obs_states[2]
                behaviour_obs2 = self.behaviour_obs_states[0] 
                
            elif action_player2 == 'keep':
                choice_obs1 = 'keep'
                reward_obs1 = self.reward_obs_states[1]
                behaviour_obs1 = self.behaviour_obs_states[1]  
                
                choice_obs2 = 'keep'
                reward_obs2 = self.reward_obs_states[0]
                behaviour_obs2 = self.behaviour_obs_states[0] 
           
            else:
                raise Exception(f'P1: {self.player1_choice} P2: {action_player2} is not a valid squence of moves!')
        
        print(f'P1: {choice_obs1} - {reward_obs1} reward, observed {behaviour_obs1} behaviour')
        print(f'P2: {choice_obs2} - {reward_obs2} reward, observed {behaviour_obs2} behaviour')
        obs1 = [reward_obs1, behaviour_obs1, choice_obs1]
        obs2 = [reward_obs2, behaviour_obs2, choice_obs2]           
        
        return [obs1, obs2]
                
                
            
            
    
            
            
            
        
        
    
