from scipy.special import softmax
import os, sys, glob
import numpy as np
root = os.path.dirname(os.getcwd())
sys.path.append(root)
from gms import GenerativeModel


def softmax_params(*params):

    """
    Use this when you want to plot the params. 
    """

    (p_share_f, p_share_h, p_share_r,
    p_ff, p_fh, p_hf, p_hh, p_rf, p_rh,
    p_r0, p_r1, p_r2,
    pr_context_pos, pr_context_neg) = params

    # softmax A
    p_share_f_s, p_share_h_s, p_share_r_s = softmax([p_share_f, p_share_h, p_share_r])
    # softmax B
    p_ff_s, p_fh_s = softmax([p_ff, p_fh])
    p_hf_s, p_hh_s = softmax([p_hf, p_hh])
    p_rf_s, p_rh_s = softmax([p_rf, p_rh])

    # softmax C
    p_r0_s, p_r1_s, p_r2_s = softmax([p_r0, p_r1, p_r2])

    # softmax D
    pr_pos_s, pr_neg_s, pr_rand_s = softmax([pr_context_pos, pr_context_neg, 0])

    softmaxed = {'A': [p_share_f_s, p_share_h_s, p_share_r_s],
                 'B': [p_ff_s, p_fh_s, p_hf_s, p_hh_s, p_rf_s, p_rh_s],
                 'C': [p_r0_s, p_r1_s, p_r2_s],
                 'D': [pr_pos_s, pr_neg_s, pr_rand_s]}

    return softmaxed


def make_soft_A(p_share_friendly, p_share_hostile, p_share_random):

    A = np.zeros([1,3,3,3])


    return soft_A


def make_soft_B(p_ff, p_fh, p_hf, p_hh, p_rf, p_rh):
        
    B_context = np.zeros([3, 3, 1])

    line1 = softmax([p_ff, p_fh, 1-(p_ff + p_fh)])
    line2 = softmax([p_hf, p_hh, 1-(p_hf + p_hh)])
    line3 = softmax([p_rf, p_rh, 1-(p_rf + p_rh)])
        
    #transition from friendly to...
    B_context[ : , 0, 0] = line1
    #transition away from hostile to
    B_context[ : , 1, 0] = line2
    #transition from random to...
    B_context[ : , 2, 0] = line3

    soft_B = B_context    

    return soft_B


if __name__ == '__main__':

    params = [0.5000000407824076,
             0.4999999592175924,
             0.49999998993726746,
             0.5000000100627325,
             0.5000001544075471,
             0.49999984559245286]

    soft_B = make_soft_B(*params)

    
