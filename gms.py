# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:16:14 2022

@author: janik and annae

"""

import numpy as np
from pymdp import utils
from envs import TrustGame
import pdb
from scipy.special import softmax

class GenerativeModel(TrustGame):
    """ Trust Game Generative Model

    :param MyEnv: Environment class
    :param p_context_friendly: If Agent beliefs in a friendly context how likely does he expect the other player to share with him. The default is 0.8.
    :type p_context_friendly: TYPE, optional
    :param p_context_hostile: If Agent beliefs himself in a hostile context how probable does he expect the other player to share with him. The default is 0.2.
    :type p_context_hostile: TYPE, optional
    :param p_context_random: If Agent beliefs himself in a random context how probable does he expect the other player to share with him. The default is 0.5.
    :type p_context_random: TYPE, optional

    """
    def __init__(self, p_share_friendly = 0.8 , p_share_hostile = 0.2, p_share_random = 0.5 ):
        """

        :param MyEnv: type MyEnv: Environment class
        :param p_context_friendly: If Agent beliefs in a friendly context how likely does he expect the other player to share with him. The default is 0.8.
        :type p_context_friendly: TYPE, optional
        :param p_context_hostile: If Agent beliefs himself in a hostile context how probable does he expect the other player to share with him. The default is 0.2.
        :type p_context_hostile: TYPE, optional
        :param p_context_random: If Agent beliefs himself in a random context how probable does he expect the other player to share with him. The default is 0.5.
        :type p_context_random: TYPE, optional

        """
        super().__init__()
        self.p_share_friendly = p_share_friendly
        self.p_share_hostile = p_share_hostile
        self.p_share_random = p_share_random
        self.gen_ABCD()
    
    
    def gen_ABCD(self):
        """ """
        self.gen_A()
        self.gen_B()
        self.gen_C()
        self.gen_D()
        
    
    def gen_A(self,p_share_friendly=None, p_share_hostile=None, p_share_random=None):

        # set beliefs of observing a return of investment in each context
        if p_share_friendly is not None:
            self.p_share_friendly = p_share_friendly
        if p_share_hostile is not None:
            self.p_share_hostile = p_share_hostile
        if p_share_random is not None:
            self.p_share_random = p_share_random
        
        """Generate the A array"""
        A = utils.obj_array( self.num_modalities )
        
        # fill out reward observation modality A_reward
        # mapping between reward and context
        A_reward = np.zeros((len(self.reward_obs_states), len(self.context_states), len(self.choice_states)))
        for choice_id, choice_name in enumerate(self.choice_states):
            
            if choice_name == 'keep':
                A_reward[2, : , choice_id] = 1.0                
                # remember: reward modality = [1.0, 0.0, 0.5] (==self.reward_obs_states) ;
                # choice_states = ['share', 'keep', 'start']

            elif choice_name == 'share':
                
                #remember: context_states = ['friendly', 'hostile', 'random']
                #1 context = friendly
                A_reward[0 , 0, choice_id] = self.p_share_friendly
                A_reward[1 , 0, choice_id] = 1 - self.p_share_friendly
                #2 context hostile
                A_reward[0, 1, choice_id] = self.p_share_hostile
                A_reward[1, 1, choice_id] = 1 - self.p_share_hostile
                if self.neutral_hstate is True:
                    A_reward[0, 2, choice_id] = self.p_share_random
                    A_reward[1, 2, choice_id] = 1 - self.p_share_random
            

            elif choice_name == 'start':
                A_reward[1, : , choice_id] = 1.0

        A[0] = softmax(A_reward, axis=0)


        ######### behaviour observation modality #########
        # This is the mapping between agent's actions and expected reward... 
        A_behaviour = np.zeros((len(self.behaviour_obs_states), len(self.context_states), len(self.choice_states)))
        
        for choice_id, choice_name in enumerate(self.choice_states):
            #remember behaviour obs modality: ['social', 'anti-social', 'unknown']
            #remember: context_states = ['friendly', 'hostile', 'random']
            if choice_name == 'keep':
                A_behaviour[2, : , choice_id] = 1.0 #agent expects 'unknown' if he uses action 'keep'
                        
            elif choice_name == 'share':
                #A_behaviour[0, : , choice_id] = 0
                #1 friendly context: expectation of action consequences
                A_behaviour[0, 0 , choice_id] = self.p_share_friendly
                A_behaviour[1, 0 , choice_id] = 1 - self.p_share_friendly
                #2 hostile
                A_behaviour[0, 1 , choice_id] = self.p_share_hostile #0.2
                A_behaviour[1, 1 , choice_id] = 1 - self.p_share_hostile
                if self.neutral_hstate is True:
                    A_behaviour[0, 2, choice_id] = self.p_share_random
                    A_behaviour[1, 2, choice_id] = 1 - self.p_share_random
                    
                    
            elif choice_name == 'start':
                A_behaviour[2 , : , choice_id] = 1.0
                 
                
        A[1] = softmax(A_behaviour, axis=0)

        #########
        # choice observation modality
        # this is the mapping between sensed states and true states.
        # this mapping is deterministic for all agent types. 
        A_choice = np.zeros((len(self.choice_obs_states), len(self.context_states), len(self.choice_states)))
        for choice_id in range(len(self.choice_obs_states)):
            A_choice[choice_id, : , choice_id] = 1.0
        
        A[2] = softmax(A_choice, axis=0)
        
        self.A = A


    def gen_B_opt(self, p_ff, p_fh, p_hf, p_hh, p_rf, p_rh):

        """
        meaning of args:
        p_ff: prob friendly-friendly transition, p_fh: friendly-hostile transition...
        p_hf: prob hostile-friendly p_hh: prob hostile-hostile
        p_rf: prob random-friendly, p_rh: prob random-hostile
        """

        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))

        # need to softmax here I think!
        line1 = softmax([p_ff, p_fh, 1-(p_ff + p_fh)])
        line2 = softmax([p_hf, p_hh, 1-(p_hf + p_hh)])
        line3 = softmax([p_rf, p_rh, 1-(p_rf + p_rh)])
        
        #transition from friendly to...
        B_context[ : , 0, 0] = line1
        #transition away from hostile to
        B_context[ : , 1, 0] = line2
        #transition from random to...
        B_context[ : , 2, 0] = line3
                
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))

        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share', 'start', 'observe']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B

        
        
        
    def gen_B(self):
        """ """

        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))
        
        if self.neutral_hstate is True:
            #transition from friendly to...
            B_context[ : , 0, 0] = [0.9, 0.02, 0.08]
            #transition away from hostile to
            B_context[ : , 1, 0] = [0.32, 0.6, 0.08]
            #transition from random to...
            B_context[ : , 2, 0] = [0.5, 0.3, 0.2]
        else:
            #friendly
            B_context[0, 0, 0] = 0.98 # friendly to friendly
            B_context[1, 0, 0] = 0.02 # friendly to hostile
            B_context[0, 1, 0] = 0.02 # hostile to hostile
            B_context[1, 1, 0] = 0.98 # hostile to friendly ???
        #B_context[ : , : , 0] = np.eye(len(self.context_states))        
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))

        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share', 'start', 'observe']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B

    def gen_depressedB(self):
        """ Fixed B dynamics over time """
        
        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile','random']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))
        
        if self.neutral_hstate is True:
            #transition from cooperative to...
            B_context[ : , 0, 0] = [0.20, 0.70, 0.1] #0.50, 0.40 before
            #transition from hostile to
            B_context[ : , 1, 0] = [0.05, 0.9, 0.05]
            #transition from random to...
            B_context[ : , 2, 0] = [0.15, 0.15, 0.7]
        else:
            #friendly
            B_context[0, 0, 0] = 0.95   # friendly - friendly
            B_context[1, 0, 0] = 0.05   # hostile - friendly
            B_context[0, 1, 0] = 0.15   # friendly - hostile
            B_context[1, 1, 0] = 0.85   # hostile - hostile
        #B_context[ : , : , 0] = np.eye(len(self.context_states))        
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))
        
        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B


    def gen_insecureB(self):
        """ Fixed B dynamics over time """
        
        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile','random']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))
        
        #transition from friendly to...
        B_context[ : , 0, 0] = [0.2, 0.3, 0.50]
        #transition from hostile to
        B_context[ : , 1, 0] = [0.1, 0.4, 0.50]
        #transition from random to...
        B_context[ : , 2, 0] = [0.2, 0.3, 0.50]
                
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))
        
        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B
        
    def gen_staticB(self):
        """ Fixed B dynamics over time """
        
        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile','random']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))
        
        if self.neutral_hstate is True:
            #transition from friendly to...
            B_context[ : , 0, 0] = [0.6, 0.3, 0.1]
            #transition away from hostile to
            B_context[ : , 1, 0] = [0.05, 0.8, 0.15]
            #transition from random to...
            B_context[ : , 2, 0] = [0.1, 0.6, 0.3]
        else:
            #friendly
            B_context[0, 0, 0] = 0.9   # friendly - friendly
            B_context[1, 0, 0] = 0.0  # hostile - friendly
            B_context[0, 1, 0] = 0.1  # friendly - hostile
            B_context[1, 1, 0] = 1.0   # hostile - hostile
        #B_context[ : , : , 0] = np.eye(len(self.context_states))        
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))
        
        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B

    def gen_defeatedB(self):
        """ Fixed B dynamics over time """
        
        B = utils.obj_array(self.num_hfactors)
        # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
        #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile','random']
        B_context = np.zeros((len(self.context_states), len(self.context_states), len(self.context_action_names)))
        
        if self.neutral_hstate is True:
            #transition from cooperative to...
            B_context[ : , 0, 0] = [0.20, 0.60, 0.2] #0.50, 0.40 before
            #transition from hostile to
            B_context[ : , 1, 0] = [0.15, 0.8, 0.05]
            #transition from random to...
            B_context[ : , 2, 0] = [0.2, 0.3, 0.5]
        else:
            #friendly
            B_context[0, 0, 0] = 0.9   # friendly - friendly
            B_context[1, 0, 0] = 0.0  # hostile - friendly
            B_context[0, 1, 0] = 0.1  # friendly - hostile
            B_context[1, 1, 0] = 1.0   # hostile - hostile
        #B_context[ : , : , 0] = np.eye(len(self.context_states))        
        B[0] = B_context
        
        #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
        B_choice = np.zeros((len(self.choice_states), len(self.choice_states), len(self.choice_action_names)))
        
        for action_id, choice_action_name in enumerate(self.choice_action_names):
            #remember: choice_action_names = ['keep', 'share']
            B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                        #could model loss of control over your own actions ?
        B[1] = B_choice
        self.B = B
        
    def gen_C(self, p_r0 = 1.2, p_r1 = -3.0, p_r2 = 1.0):
        """Generate the C Array (Prior Preferences)

        :param p_r0: log pref for reward = "win" (1.0/1.5). (Default value = 1.2)
        :param p_r1: log pref for reward = "loss" (0/0). (Default value = -3.0)
        :param p_r2: log pref for reward = "keep" (0.5/0.5). (Default value = 1.0)
        """
        
        """
        One could add a preference for social behaviour of partner because then hes going to share. maybe thats the missing link to produce reasonable behaviour and correct inference?
        """
        C = utils.obj_array_zeros(self.num_obs) 
        
        #C[0] reward, C[1] behaviour, C[2] choice
        # prior reward preferences
        #remember: reward_obs_states = [0.5, 0.0, 1.0]
        C_reward = np.zeros(len(self.reward_obs_states))
        
        C_reward[:3] = softmax([p_r0, p_r1, p_r2])
    
        C[0] = C_reward
        self.C=C
        
        
    def gen_D(self, pr_context_pos = 0.5, pr_context_neg = 0.5):
        """Generate the D Vectors (Prior beliefs over hidden states)

        :param p_context_pos: Default value = 0.5)
        :param p_context_neg: Default value = 0.2)
        :param p_context_ran: Default value = 0.3)

        """
        D = utils.obj_array(self.num_hfactors)
        
        if self.neutral_hstate is True:
            #D_context = [pr_context_pos, pr_context_neg, 1.0 - pr_context_pos - pr_context_neg] # simulations
            D_context = [pr_context_pos, pr_context_neg, 0]
            D_context = softmax(D_context)
        else:
            D_context = [pr_context_pos, pr_context_neg]
            
        D[0] = np.array(D_context)
        
        D_choice = np.zeros(len(self.choice_states))
        #agent starts in start position
        D_choice[2] = 1.0

        D[1] = D_choice
        
        #print(f'Beliefs about context: {D[0]}')
        #print(f'Beliefs about starting action: {D[1]}')
              
        self.D = D



def gm_trust_agent(MyEnv, p_context_friendly = 0.8 , p_context_hostile = 0.2 , p_context_random = 0.5):
    """

    :param MyEnv: type MyEnv: Environment class
    :param p_context_friendly: If Agent beliefs in a friendly context how likely does he expect the other player to share with him. The default is 0.8.
    :type p_context_friendly: TYPE
    :param p_context_hostile: If Agent beliefs himself in a hostile context how probable does he expect the other player to share with him. The default is 0.2.
    :type p_context_hostile: TYPE
    :param p_context_random: If Agent beliefs himself in a random context how probable does he expect the other player to share with him. The default is 0.5.
    :type p_context_random: TYPE

    """
    
    """ Generate the A array """
    A = utils.obj_array( MyEnv.num_modalities )
    
    # fill out reward observation modality A_rew
    A_reward = np.zeros((len(MyEnv.reward_obs_states), len(MyEnv.context_states), len(MyEnv.choice_states)))
    
    for choice_id, choice_name in enumerate(MyEnv.choice_states):
        if choice_name == 'keep':
            #remember: reward modality = [0.5, 0, 1.0] (==MyEnv.reward_obs_states)
            A_reward[0, : , choice_id] = 1.0
        elif choice_name == 'share':
            #remember: context_states = ['friendly', 'hostile', 'random']
            #1 context = friendly
            A_reward[1, 0, choice_id] = 1.0 - p_context_friendly
            A_reward[2, 0, choice_id] = p_context_friendly
            #2 context hostile
            A_reward[1, 1, choice_id] = 1.0 - p_context_hostile
            A_reward[2, 1, choice_id] = p_context_hostile
            #3 context random
            A_reward[1, 2, choice_id] = 1.0 - p_context_random
            A_reward[2, 2, choice_id] = p_context_random
        elif choice_name == 'start':
            A_reward[1, : , choice_id] = 1.0
              
    A[0] = A_reward
    
    # behaviour observation modality 
    A_behaviour = np.zeros((len(MyEnv.behaviour_obs_states), len(MyEnv.context_states), len(MyEnv.choice_states)))
    
    for choice_id, choice_name in enumerate(MyEnv.choice_states):
        #remember behaviour obs modality: ['unknown', 'anti-social', 'social']
        
        if choice_name == 'keep':
            A_behaviour[0, : , choice_id] = 1.0 #agent expects 'unknown' if he uses action 'keep'
        elif choice_name == 'share':            #could be interesting to alter
            #A_behaviour[0, : , choice_id] = 0
            #1 friendly
            A_behaviour[1, 0 , choice_id] = 1.0 - p_context_friendly
            A_behaviour[2, 0 , choice_id] = p_context_friendly
            #2 hostile
            A_behaviour[1, 1 , choice_id] = 1.0 - p_context_friendly
            A_behaviour[2, 1 , choice_id] = p_context_friendly
            #3 random
            A_behaviour[1, 2 , choice_id] = 1.0 - p_context_friendly
            A_behaviour[2, 2 , choice_id] = p_context_friendly
        elif choice_name == 'start':
            A_behaviour[ 0 , : , choice_id] = 1.0
            
    A[1] = A_behaviour
    
    # choice observation modality
    A_choice = np.zeros((len(MyEnv.choice_obs_states), len(MyEnv.context_states), len(MyEnv.choice_states)))
    for choice_id in range(len(MyEnv.choice_obs_states)):
        A_choice[choice_id, : , choice_id] = 1.0
    
    A[2] = A_choice 
    
    
    """ Generate the B Array """
    B = utils.obj_array(MyEnv.num_hfactors)
    
    # context state dynamics """ Fill out the context state factor dynamics, a sub-array of `B` which we'll call `B_context`"""
    #remember: context_action_names = ['do nothing'] and context_states = ['friendly','hostile','random']
    B_context = np.zeros((len(MyEnv.context_states), len(MyEnv.context_states), len(MyEnv.context_action_names)))
    B_context[ : , : , 0] = np.eye(len(MyEnv.context_states))
    
    B[0] = B_context
    
    #choice state dynamics """Fill out the choice factor dynamics, a sub-array of `B` which we'll call `B_choice`"""
    B_choice = np.zeros((len(MyEnv.choice_states), len(MyEnv.choice_states), len(MyEnv.choice_action_names)))
    
    for action_id, choice_action_name in enumerate(MyEnv.choice_action_names):
        #remember: choice_action_names = ['keep', 'share']
        B_choice[action_id, : , action_id] = 1.0 #change this to add 'uncertainty' about actions -> agent is not 100% sure what action_state hes going to get if he chooses a certain action
                                                    #could model loss of control over your own actions ?
    B[1] = B_choice
       

    """ Generate the C Array (Prior Preferences) """
    C = utils.obj_array_zeros(MyEnv.num_obs) 
    
    #C[0] reward, C[1] behaviour, C[2] choice
    # prior reward preferences
    #remember: reward_obs_states = [0.5, 0.0, 1.0]
    C_reward = np.zeros(len(MyEnv.reward_obs_states))
    C_reward[0] = 0.1
    C_reward[1] = -1.0 
    C_reward[2] = 3.0
    
    C[0] = C_reward
    
    """ Generate the D Vectors (Prior beliefs over hidden states) """
    D = utils.obj_array(MyEnv.num_hfactors)
    
    D_context = np.array([0.5,0.2,0.3])
    D[0] = D_context
    
    D_choice = np.zeros(len(MyEnv.choice_states))
    #agent starts in start position
    D_choice[2] = 1.0

    D[1] = D_choice
    
    print(f'Beliefs about context: {D[0]}')
    print(f'Beliefs about starting action: {D[1]}')
          
    return A, B, C, D
