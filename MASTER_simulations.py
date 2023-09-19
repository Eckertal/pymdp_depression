"""
MASTER SCRIPT!

This master script can be used to reproduce all simulations with one sweep.

1. Transdiagnostic Mechanisms
- simulation to show the effects of perturbations to specific parameters A,B,C or D.

2. Within disorders simulations
- shows 2 types of depression.

3. Across disorders
- Simulate an agent reminiscent of social phobia.

author @annae

"""

from envs import TrustGame
from library import get_player_agent
from inference import run_active_inference_loop
from vis import *
from main_simulation_func import run_simulation

import random

random.seed(10)

# timestep = 60

# choose plotting option: 'series_only', 'all', 'none'
plotting= 'series_only'

"""
TRANSDIAGNOSTIC MECHANISMS ----------------------
"""

# A matrix
run_simulation('hostile','biased_A', plotting=plotting)
run_simulation('friendly', 'biased_A', plotting=plotting)

# B matrix
# use gen_depressed B
run_simulation('hostile', 'biased_B', plotting=plotting)
run_simulation('friendly', 'biased_B', plotting=plotting)

# C Matrix
run_simulation('hostile', 'biased_C', plotting=plotting)
run_simulation('friendly', 'biased_C', plotting=plotting)
            

# D Matrix
run_simulation('hostile', 'biased_D', plotting=plotting)
run_simulation('friendly', 'biased_D', plotting=plotting)


"""
WITHIN-DISORDER SIMULATIONS ----------------------
"""

# healthy person
run_simulation('hostile', 'Player1_healthy', plotting=plotting)
run_simulation('friendly', 'Player1_healthy', plotting=plotting)

# Depressed agents!
# Type 1 depressed patient: perturbations esp. in C and D parameters
run_simulation('hostile', 'Type1_depressed', plotting=plotting)
run_simulation('friendly', 'Type1_depressed', plotting=plotting)


# Type 2 depressed patient: perturbations esp. in B and D parameter
run_simulation('hostile', 'Type2_depressed', plotting=plotting)
run_simulation('friendly', 'Type2_depressed', plotting=plotting)

"""
#ACROSS-DISORDER SIMULATIONS ----------------------
"""

# Social phobia agents. 
# Type 1 SoPho: lossAvoidance and high prior context uncertainty
run_simulation('hostile', 'Type1_social_phobia', plotting=plotting)
run_simulation('friendly', 'Type1_social_phobia', plotting=plotting)

# Type 2 SoPho: loss avoidance and otherwise normal
run_simulation('hostile', 'Type2_social_phobia', plotting=plotting)
run_simulation('friendly', 'Type2_social_phobia', plotting=plotting)

