# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:55:41 2022

@author: janik

Active List:
    - Integrate new mode into generative model class


"""

def run_active_inference_loop(MyAgent, Env, T = 5):

    """Initialize the first observation
    :param MyAgent: 
    :param Env: 
    :param T:  (Default value = 5)
    """
    obs_label = [0, 'unknown', "start"]  # agent observes itself getting a "0" reward, observing "unknown" behaviour, and in the `start` location
    obs = [Env.reward_obs_states.index(obs_label[0]), Env.behaviour_obs_states.index(obs_label[1]), Env.choice_obs_states.index(obs_label[2])]

    for t in range(T):
        

        qs = MyAgent.infer_states(obs)
        MyAgent.beliefs_context.append(qs[0])
        #imp.plot_beliefs(qs[0], title_str = f"Beliefs about the context at time {t}")
        q_pi, efe = MyAgent.infer_policies()
        #print(f'qs: {qs}\n q_pi: {q_pi}\nqs[0]: {qs[0]}\n')
        chosen_action_id = MyAgent.sample_action()
        
        movement_id = int(chosen_action_id[1])
        
        choice_action = Env.choice_action_names[movement_id]
        
        obs_label = Env.step(choice_action)
        
        obs = [Env.reward_obs_states.index(obs_label[0]), Env.behaviour_obs_states.index(obs_label[1]), Env.choice_obs_states.index(obs_label[2])]
     
        if MyAgent.updateA:
            MyAgent.update_A(obs)
        if MyAgent.updateB:
                MyAgent.update_B(qs) 
        if MyAgent.updateD:
            MyAgent.update_D(qs) #macht das einen Unterschied? 
        #store partner behaviour
        MyAgent.partner_behaviour.append(obs_label[1])           
        print(f'Beliefs about context at time {t}: {MyAgent.qs[0]}')
        print(f'Action at time {t}: {choice_action}')
        print(f'Reward at time {t}: {obs_label[0]}')
        print(f'Observed action at time {t}: {obs_label[1]}')
        
def run_active_inference_loop_coop(Agent1, Agent2, Env, T = 5, updateA = True, updateB = True, updateD = True):

    """Initialize the first observation
    :param MyAgent: 
    :param Env: 
    :param T:  (Default value = 5)
    """
    obs_label1 = [0, 'unknown', "start"]  # agents observe themselves getting a "0" reward, observing "unknown" behaviour, and in the `start` location
    obs_label2 = [0, 'unknown', "start"]
    obs1 = [Env.reward_obs_states.index(obs_label1[0]), Env.behaviour_obs_states.index(obs_label1[1]), Env.choice_obs_states.index(obs_label1[2])]
    obs2 = [Env.reward_obs_states.index(obs_label2[0]), Env.behaviour_obs_states.index(obs_label2[1]), Env.choice_obs_states.index(obs_label2[2])]
    obs = [obs1,obs2]
    
    for t in range(T):
        choice_actions=[]
        qs_list = []
        for MyAgent, myobs in zip([Agent1,Agent2], obs):
            qs = MyAgent.infer_states(myobs)
            qs_list.append(qs)
            MyAgent.beliefs_context.append(qs[0])
            q_pi, efe = MyAgent.infer_policies()
            chosen_action_id = MyAgent.sample_action()
            movement_id = int(chosen_action_id[1])
            choice_action = Env.choice_action_names[movement_id]
            choice_actions.append(choice_action)

        
        obs_labels = Env.step(choice_actions)
        obs = []
        for obs_label in obs_labels:
            obsx = [Env.reward_obs_states.index(obs_label[0]), Env.behaviour_obs_states.index(obs_label[1]), Env.choice_obs_states.index(obs_label[2])]
            obs.append(obsx)        
        
        for idx, MyAgent in enumerate([Agent1,Agent2]):
            qs = qs_list[idx]
            obsx = obs[idx]
            choice_action = choice_actions[idx]
            obs_label = obs_labels[idx]
            
            if MyAgent.updateA:
                MyAgent.update_A(obsx)
            if MyAgent.updateB:
                    MyAgent.update_B(qs) 
            if MyAgent.updateD:
                MyAgent.update_D(qs) #macht das einen Unterschied? 
            #store partner behaviour
            MyAgent.partner_behaviour.append(obs_label[1])
            
            print(f'Player {idx+1} ({MyAgent.name}): \n')
            print(f'Beliefs about context at time {t}: {MyAgent.qs[0]}')
            print(f'Action at time {t}: {choice_action}')
            print(f'Reward at time {t}: {obs_label[0]}')
            print(f'Observed action at time {t}: {obs_label[1]}')
            print('\n')
            
def run_active_inference_loop_coop_asymmetric(Agent1, Agent2, Env, T = 5, updateA = True, updateB = True, updateD = True):

    """Initialize the first observation
    :param MyAgent: 
    :param Env: 
    :param T:  (Default value = 5)
    """
    # ------- Start observation P1 -------
    obs_label1 = ['nothing', 'unknown', 'start']  # agents observe themselves getting a "0" reward, observing "unknown" behaviour, and in the `start` location
    obs1 = [Env.reward_obs_states.index(obs_label1[0]), Env.behaviour_obs_states.index(obs_label1[1]), Env.choice_obs_states.index(obs_label1[2])]
    
    for t in range(T):   
        # ------- P1 beliefs update + P1 sample action -------
        qs1 = Agent1.infer_states(obs1)
        Agent1.beliefs_context.append(qs1[0])
        q_pi1, efe1 = Agent1.infer_policies()
        chosen_action_id1 = Agent1.sample_action()
        movement_id1 = int(chosen_action_id1[1])
        choice_action1 = Env.choice_action_names[movement_id1]
        
        # ------- P1 action -------
        player1_trusts =  Env.player1_action(choice_action1)
        
        # ------- P2 observation and beliefs update -------
        obs_label2 = Env.player2_observation()
        obs2 = [Env.reward_obs_states.index(obs_label2[0]), Env.behaviour_obs_states.index(obs_label2[1]), Env.choice_obs_states.index(obs_label2[2])]
        qs2 = Agent2.infer_states(obs2)
        Agent2.beliefs_context.append(qs2[0])
        update_gm(Agent2, obs = obs2)
        if player1_trusts == True:
            # ------- P2 sample action -------
            print('P1 trusted - sample action of P2...')
            q_pi2, efe2 = Agent2.infer_policies()
            chosen_action_id2 = Agent2.sample_action()
            movement_id2 = int(chosen_action_id2[1])
            choice_action2 = Env.choice_action_names[movement_id2] 
            print(f'chocie_action: {choice_action2}')   
        else:
            choice_action2 = 'nothing-to-do'

        # ------- Environment step -------
        obs_label1, obs_label2 = Env.step(choice_action2)
        obs1 = [Env.reward_obs_states.index(obs_label1[0]), Env.behaviour_obs_states.index(obs_label1[1]), Env.choice_obs_states.index(obs_label1[2])]
        obs2 = [Env.reward_obs_states.index(obs_label2[0]), Env.behaviour_obs_states.index(obs_label2[1]), Env.choice_obs_states.index(obs_label2[2])]
        update_gm(Agent1, obs1, qs1)
        if player1_trusts:
            
            # ------- P2 infer states again after second observation -------
            qs2 = Agent2.infer_states(obs2)     
            update_gm(Agent2, obs2, qs2)
        # ------- P1, P2 update generative model -------

                
        # ------ store partner behaviour -------
        Agent1.partner_behaviour.append(obs_label1[1])
        Agent2.partner_behaviour.append(obs_label2[1])
"""            
            # ------ print time step -------
            print(f'Player {idx+1} ({MyAgent.name}): \n')
            print(f'Beliefs about context at time {t}: {MyAgent.qs[0]}')
            print(f'Action at time {t}: {choice_action}')
            print(f'Reward at time {t}: {obs_label[0]}')
            print(f'Observed action at time {t}: {obs_label[1]}')
            print('\n')
"""            
def update_gm(MyAgent, obs=False, qs = False):
            if obs:
                if MyAgent.updateA:
                    MyAgent.update_A(obs)
            if qs is not False:
                if MyAgent.updateB:
                        MyAgent.update_B(qs) 
                if MyAgent.updateD:
                    MyAgent.update_D(qs)