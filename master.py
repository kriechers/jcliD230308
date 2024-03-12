import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from os import mkdir
from os.path import isdir, exists
from bifurcations_v6_1 import compute_bifurcation_diagram
from deterministic_trajectory import compute_det_traj
from stochastic_trajectory import compute_ran_traj
from transient_trajectories import compute_trans_traj
#from waiting_time_stats import compute_duration_stats
from miscellanious import dirname
from fig01 import create_fig01
from figA1 import plot_atm_ocean_bifurcation
from figA2 import plot_atm_ice_bifurcation
from fig04 import plot_det_trajs
from fig05 import plot_ran_traj
from fig06 import plot_trans_traj
from matplotlib import rc
import tol_colors as tc

print('master running')

#################################################################
# set matplotlib parameters                                     #
#################################################################

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif',
                     'font.size': 10,
                     'axes.labelsize': 12,
                     'axes.titlesize': 14,
                     'figure.titlesize': 16})

plt.rc('text.latex', preamble=(r'\usepackage{amsmath}' +
                               r'\usepackage{wasysym}' +
                               r'\usepackage{xcolor}' +
                               r'\usepackage{textcomp}'))

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

# for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)


''' 
This file is used to globally set the model parameters and
execute different subroutines to compute all results presented in
manuscript 'Glacial abrupt climate change as a multi-scale
phenomenon resulting from monostable excitable dynamics' 
(Riechers, Gottwald, Boers, 2023). 
'''

# set the random seed for reproducability
np.random.seed(seed=2)

#################################################################
# Paramters                                                     #
#################################################################

params = {'theta0': 1.3,
          'tau_ocean': 800,
          'tau_seaice': 200,
          'tau_atm': 0.6,
          'k': 0.62,
          'theta_max': 2,
          'theta_min': 1.29,
          'sigma_laminar': 2,
          'sigma_turbulent': 0.01,  # multiplied with dt
          'sigma_interstadial': 0.006,  # multiplied with sqrt(dt)
          'sigma_theta': 1,
          'obs_laminar': 0.2,  #first 11 entries used for dirname
          'eta': 4, #eta =7 yields way to short GI 
          'bias_correction': -1.3076202432497247e-07,
          'L0': 1.75,
          'L1': 1.85,
          'L2': 0.35,
          'Delta': 0.25,
          'ha': 0.08,
          'R0': 0.4,
          'mu': 7.5,
          'sigma': 0.7,
          'xi': 1,
          'gamma0': 0.5,
          'Delta_gamma': 3.5,
          'omega': 0.8,
          'I0': -0.5}

# ----------------------------------------------------------------

if not isdir('output'):
    mkdir('output')

figdir = dirname(params, sub='figures')
resdir = dirname(params, sub='results')
base = figdir.split('/')[0] + '/' + figdir.split('/')[1]
if not isdir(base):
    mkdir(base)
if not isdir(figdir):
    mkdir(figdir)
if not isdir(resdir):
    mkdir(resdir)

#################################################################
# create fig A1                                                 #
#################################################################

# compute_bifurcation_diagram(params, recompute=True)
# plot_atm_ocean_bifurcation(params)

#################################################################
# create fig A2                                                 #
#################################################################

#plot_atm_ice_bifurcation(params, (params['theta0'], 1.6), recompute = False)

#################################################################
# create fig04                                                  #
#################################################################

# perturbations = np.array([0.2, 0, -0.2, -0.5, -1, -2])
# thetas = (params['theta0'], 1.6)
# plot_det_trajs(params, perturbations, thetas, recompute = True)

#################################################################
# create fig05                                                  #
#################################################################

# for n in range(10):
#     plot_ran_traj(params, (params['theta0'], 1.6), ntraj=n, recompute=True)

#################################################################
# compute transient trajectories                                #
#################################################################

# print('computing transient trajectories')
# for n in range(100):
#     compute_trans_traj(params, ntraj=n, recompute=False)

#################################################################
# create fig06 for selected transient trajectories              #
#################################################################

plot_trans_traj(params, ntraj=0, recompute=False)

#################################################################
# create fig01                                                  #
#################################################################

create_fig01(params, ntraj=0)



