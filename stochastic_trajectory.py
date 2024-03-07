import numpy as np
import pandas as pd
from miscellanious import check_results
from miscellanious import dirname
from miscellanious import cheby_lowpass_filter
from os.path import exists
from bifurcations_v6_1 import compute_bifurcation_diagram
from full_model import deterministic_drift as drift
from scipy.stats import genpareto as gprnd



def compute_ran_traj(params, theta0, ntraj=1, recompute=False):

    ### noise parameters ####

    sigma_laminar = params['sigma_laminar']
    sigma_turbulent = params['sigma_turbulent']
    sigma_interstadial = params['sigma_interstadial']
    sigma_theta = params['sigma_theta']
    obs_laminar = params['obs_laminar']
    bias_correction = params['bias_correction']
    
    k = params['k']
    theta = sigma_laminar/k
    mean_xi = theta + sigma_laminar/(1-k)

    params['theta0'] = theta0
    filename = 'ran_traj_n%i.csv' % (ntraj)
    traj_name, traj_exists = check_results(params, filename)

    if traj_exists:
        traj = pd.read_csv(traj_name)
        print('read trajectory from ' + traj_name)

    if not traj_exists or recompute:
        directory = dirname(params)

        if not exists(directory + '/system_fp.csv'):
            compute_bifurcation_diagram(params)
        x0 = pd.read_csv(directory + '/system_fp.csv').values.squeeze()

        # transform the T,q representation into T,S representation
        x0[3] = x0[2] - x0[3]

        # initialize the system in an interstadial state
        # x0 = np.array([-0.5, 0.8, 0.5, 0.3])

        dt = 0.01
        time = np.arange(0, 30000, dt)
        ran_traj = np.zeros((len(time), 4))
        ran_traj[0] = x0
        noise = np.zeros(len(time))
        laminar_duration_list = []
        turbulent_duration_list = []
        laminar = False  # start in turbulent conditions
        print('begin Euler integration')
        for i, t in enumerate(time[:-1]):
            ran_traj[i+1] = ran_traj[i] + drift(ran_traj[i], params) * dt
            if noise[i] == 0:
                if ran_traj[i, 0] > 0.5:  # stadial conditions
                    if laminar:  # compute the noise for the next laminar period
                        n_laminar = np.ceil(gprnd.rvs(k,
                                                      loc=theta,
                                                      scale=sigma_laminar))
                        laminar = False
                        laminar_duration_list.append(n_laminar)
                        if i + n_laminar > len(time):
                            n_laminar = len(time) - i

                        # for jl in range(int(n_laminar)):
                        #    noise[i+jl] = dt*obs_laminar - bias_correction
                        noise[i:i+int(n_laminar)] = dt * \
                            obs_laminar - bias_correction

                    else:  # compute the noise for the next turbulent period
                        n_Brownian = np.ceil(mean_xi
                                             + mean_xi*(np.random.random()-0.5))
                        # n_Brownian = np.ceil(mean_xi
                        #                     + 2*(np.random.random()-0.5))
                        laminar = True
                        turbulent_duration_list.append(n_Brownian)
                        if i + n_Brownian > len(time):
                            n_Brownian = len(time) - i

                        # for jl in range(int(n_Brownian)):
                        #     noise[i+jl] = (sigma_turbulent *
                        #                    np.random.normal()-obs_laminar) * dt - bias_correction
                        noise[i:i+int(n_Brownian)] = (sigma_turbulent
                                                      * np.random.normal(size=int(n_Brownian))
                                                      - obs_laminar) * dt - bias_correction

                else:  # interstadial conditions
                    noise[i] = sigma_interstadial * \
                        np.random.normal() * np.sqrt(dt)

            ran_traj[i+1, 0] -= noise[i]
            ran_traj[i+1, 1] += sigma_theta * np.random.normal() * np.sqrt(dt)

        print('ended Euler integration')

        #########################################################
        # downsample resolution of the random trajectory to 5y  #
        # the data is first smooted with a 5 year running mean  #
        # this corresponds to physical smoothing of the d18o    #
        # more or less                                          #
        #########################################################

        w = 500
        for idx in range(ran_traj.shape[0]):
            if idx < int(w/2):
                ran_traj[idx] = np.mean(ran_traj[:idx + int(w/2)], axis=0)
            elif idx > ran_traj.shape[0]-int(w/2):
                ran_traj[idx] = np.mean(ran_traj[idx-int(w/2):], axis=0)
            else:
                ran_traj[idx] = np.mean(ran_traj[idx-int(w/2):idx + int(w/2)], axis = 0)

        res = 5
        #filtered = cheby_lowpass_filter(ran_traj.T, 1/res, 1/dt, order=4, rp=0.05)
        downsampled1 = ran_traj[::500, :]
        traj5y = pd.DataFrame(np.concatenate((time[:,None][::500],
                                              downsampled1),
                                             axis = 1),
                              columns = ['time', 'I', 'theta', 'T', 'S'])
        traj5y.to_csv(traj_name, index = False)
      
