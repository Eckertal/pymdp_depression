# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:28:54 2022

@author: janik & ale
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import matplotlib.markers as mmarkers

# functions

def plot_likelihood(matrix, title_str = "Likelihood distribution (A)"):
    """Plots a 2-D likelihood matrix as a heatmap

    :param matrix: 
    :param title_str:  (Default value = "Likelihood distribution (A)")

    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
      raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")
    
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(matrix, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
    plt.title(title_str)
    plt.show()


def plot_beliefs(belief_dist, title_str=""):
    """Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`

    :param belief_dist: 
    :param title_str:  (Default value = "")

    """

    if not np.isclose(belief_dist.sum(), 1.0):
      raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()

def plot_D_pre_post(belief_pre, belief_post, title_str=r'$D$ before and after inference'):

    fig, axs = plt.subplots(1,2)

    plt.suptitle(title_str, fontsize=20)
    # pre 
    axs[0].set_title('pre')
    axs[0].bar(range(belief_pre.shape[0]), belief_pre, color='skyblue', zorder=3)
    axs[0].set_xticks(range(belief_pre.shape[0]))

    # post
    axs[1].set_title('post')
    axs[1].bar(range(belief_post.shape[0]), belief_post, color='dodgerblue', zorder=3)
    axs[1].set_xticks(range(belief_post.shape[0]))

    plt.tight_layout()
    plt.show()


def plot_A_pre_post(A_pre, A_post, title_str = r'$A$ before and after inference'):

    A_pre_friendly = A_pre[:,0,0]
    A_pre_hostile = A_pre[:,1,0]

    A_post_friendly = A_post[:,0,0]
    A_post_hostile = A_post[:,1,0]

    fig, axs = plt.subplots(2,2)

    plt.suptitle(title_str, fontsize=20)

    axs[0][0].bar(range(A_pre_friendly.shape[0]), A_pre_friendly, color='palegreen')
    axs[0][0].set_title('pre')

    axs[0][1].bar(range(A_post_friendly.shape[0]), A_post_friendly, color='lime')
    axs[0][1].set_title('post')

    axs[1][0].bar(range(A_pre_hostile.shape[0]), A_pre_hostile, color='salmon')
    axs[1][0].set_title('pre')

    axs[1][1].bar(range(A_post_hostile.shape[0]), A_post_hostile, color='r')
    axs[1][1].set_title('post')
        
    plt.tight_layout()
    plt.show()

def plot_B_pre_post(B_pre, B_post, title_str=r'$B$ before and after inference'):

    B_pre_friendly = B_pre[:,0,0]
    B_post_friendly = B_post[:,0,0]

    B_pre_hostile = B_pre[:,1,0]
    B_post_hostile = B_post[:,1,0]

    B_pre_random = B_pre[:,2,0]
    B_post_random = B_post[:,2,0]

    fig, axs = plt.subplots(2,3)
    plt.suptitle(title_str, fontsize=20)

    row1, row2 = axs[0], axs[1]

    # PRE
    row1[0].bar(range(B_pre_friendly.shape[0]), B_pre_friendly, color='palegreen')
    row1[0].set_title('Friendly to...')

    row1[1].bar(range(B_pre_hostile.shape[0]), B_pre_hostile, color='salmon')
    row1[1].set_title('Hostile to...')

    row1[2].bar(range(B_pre_random.shape[0]), B_pre_random, color='lightgrey')
    row1[2].set_title('Random to...')

    # POST
    row2[0].bar(range(B_post_friendly.shape[0]), B_post_friendly, color='lime')

    row2[1].bar(range(B_post_hostile.shape[0]), B_post_hostile, color='r')

    row2[2].bar(range(B_post_random.shape[0]), B_post_random, color='darkgrey')

    plt.tight_layout()
    plt.show()
    
    
def dots_for_plots_func(arr):
    arr_list = []
    
    for idx in range(len(arr)-1):
        arr_list.append((arr[idx]+arr[idx+1])/2)
        
    return np.array(arr_list+[0.5])     

def colors_for_dot_plots(MyAgent,MyEnv):
    c=[]
    for action_idx, action in enumerate(MyAgent.q_pi_hist):
        m = max(action)
        idx = np.where(action==m)
        idx = idx[0]
        if idx ==1.:
            c.append('k')
        if idx==0:
            
            if MyAgent.partner_behaviour[action_idx] == 1.5:
                c.append('blue')
            else:
                c.append('red')
                
    #c= [c for action in inferred_actions with (c='blue') if >0.5 ]

    return c


def markers_for_dot_plots(MyAgent,MyEnv):
    m=[]
    for action_idx, action in enumerate(MyAgent.q_pi_hist):
        
        o = max(action)
        idx = np.where(action==o)
        idx = idx[0]
        if idx == 1.:
            m.append('X')
            
        if idx==0:
            if MyAgent.partner_behaviour[action_idx] == 1.5:
                m.append('^')
            else:
                m.append('v')

    return m


def plot_results(MyAgent, MyEnv, where_dots_idx=1):
    #idx 0 for positive , 1 for negative dots on line
    #y = [0.5]*20 
    c = colors_for_dot_plots(MyAgent, MyEnv)
    y = dots_for_plots_func(np.array(MyAgent.beliefs_context)[:,where_dots_idx]) 
    
    # ----- PLOT RESULTS -----
    plt.figure(figsize=(16,10))
    plt.title(f'{MyAgent.name}',fontsize=25)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('p(context)',fontsize=20)
    plt.xticks(np.arange(len(MyAgent.beliefs_context)),fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,1)
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,0],marker='x', c ='g', markersize=12, label='friendly')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,1],marker='x', c = 'r', markersize=12, label='hostile')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,2],marker='x', c = 'grey', markersize=12, label='unbiased')
    plt.scatter(np.arange(len(MyAgent.beliefs_context))+0.5,y,s=150,marker='o', c = c, label='action')
    plt.legend()
    plt.show()
    
def plot_results_context_change(T1, T2, c1, c2, MyAgent, MyEnv, where_dots_idx=1):
    #idx 0 for social , 1 for hostile dots on line
    #y = [0.5]*20 
    col = colors_for_dot_plots(MyAgent, MyEnv)
    
    y = dots_for_plots_func(np.array(MyAgent.beliefs_context)[:,where_dots_idx])
    
    # ----- PLOT RESULTS -----
    plt.figure(figsize=(16,10))
    plt.title(f'{MyAgent.name}',fontsize=25)
    plt.xlabel('epoch',fontsize=20)
    plt.ylabel('p(context)',fontsize=20)
    plt.xticks(np.arange(len(MyAgent.beliefs_context)),fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,1)
    
    plt.axvspan(0,T1, facecolor=c1, alpha=0.3)
    plt.axvspan(T1,T1+T2, facecolor=c2, alpha=0.3)
    
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,0],marker='x', c ='g', markersize=12, label='friendly')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,1],marker='x', c = 'r', markersize=12, label='hostile')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,2],marker='x', c = 'grey', markersize=12, label='unbiased')

    plt.scatter(np.arange(len(MyAgent.beliefs_context))+0.5,y,s=150, marker= 'o', c = col, label='action')
    plt.legend()
    plt.show()


def fancy_time_series(T1, T2, c1, c2, MyAgent, MyEnv, where_dots_idx=1, **kw):

    """
    Time series plots, but markers differ between observations
    And colors are chosen in line with recommendations for color blind ppl
    """
    
    # where_dots_idx: if == 0: on posterior line of coop, if == 1: on hostile line. 
    col = colors_for_dot_plots(MyAgent, MyEnv)
    m = markers_for_dot_plots(MyAgent, MyEnv)
    y = dots_for_plots_func(np.array(MyAgent.beliefs_context)[:,where_dots_idx])

    # basic plot features
    plt.figure(figsize=(16,10))
    plt.title(f'{MyAgent.name}',fontsize=25)
    plt.xlabel(r'$t$',fontsize=20)
    plt.ylabel(r'$p(context)$',fontsize=20)
    plt.xticks(np.arange(len(MyAgent.beliefs_context)),fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0,1)

    # background colors
    plt.axvspan(0,T1, facecolor=c1, alpha=0.3)
    plt.axvspan(T1,T1+T2, facecolor=c2, alpha=0.3)
    
    # plot lines for posterior beliefs
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,0],marker='.', c ='b', markersize=12, label='cooperative')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,1],marker='.', c = 'r', markersize=12, label='hostile')
    plt.plot(np.arange(len(MyAgent.beliefs_context)), np.array(MyAgent.beliefs_context)[:,2],marker='.', c = 'grey', markersize=12, label='random')

    # scatter chosen actions on top
    sc = plt.scatter(np.arange(len(MyAgent.beliefs_context))+0.5,y,s=150,c=col,alpha=1., label='action',**kw)
    
    if (m is not None):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
                
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)

        sc.set_paths(paths)

    plt.legend()
    plt.show()
    

def plot_earned_rewards(df, title='Rewards earned per agent type', filename='earned_rewards_per_agent.svg'):

    plt.figure(figsize=(16,10))
    plt.title(title, fontsize=25)
    sns.violinplot(data=df, palette='Spectral')
    #plt.xticks(rotation=45)
    plt.savefig(filename)
    plt.show()

