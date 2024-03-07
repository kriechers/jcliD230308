import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import exists
from scipy.interpolate import interp1d
from scipy.stats import genpareto as gprnd
from miscellanious import check_results, dirname
from miscellanious import cheby_lowpass_filter
from miscellanious import preprocess_LR04
from full_model import deterministic_drift as drift


dt = 0.01
t_i = -115000
t_f = -15000

LR04 = preprocess_LR04(t_i, t_f, dt, 40000)

time = np.arange(t_i, t_f, dt)


# control plot
# fig, ax = plt.subplots()
# ax.plot(time, theta0[:len(time)])
# ax.axhline(1.285, color = 'k')
# ax.axvline(time[idx_max])
# ax.axvline(time[idx_min])

def compute_trans_traj(params, ntraj=1, recompute=False):

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


    # define min and max for theta0
    theta0_min = params['theta_min']
    theta0_max = params['theta_max']

    # find corresponding coefficients for linear dependence on LR04
    idx_max = np.argmax(LR04[1000000:]) + 1000000
    idx_min = np.argmin(LR04[1000000:]) + 1000000

    a = (theta0_max - theta0_min)/(LR04[idx_max] -
                                   LR04[idx_min])
    b = theta0_min - a * LR04[idx_min]

    theta0 = LR04 * a + b

    params['theta0'] = 1.3  # is set to 1.3 to store transient trajectories
    directory = dirname(params)
    filename = 'transient_traj_n%i.csv' % (ntraj)
    traj_name = directory + '/trans_traj_5y_n%i.csv' % (ntraj)
    traj_exists = exists(traj_name)

    if traj_exists:
        traj = pd.read_csv(traj_name)
        print('read trajectory from ' + traj_name)

    if not traj_exists or recompute:

        #################################################################
        # initialize the system                                         #
        #################################################################
        x0 = np.array([-0.5, 0.8, 0.5, 0.3])
        xx = np.zeros((len(time), 4))
        driftx = np.zeros((len(time), 4))
        xx[0] = x0
        noise = np.zeros(len(time))
        laminar = False

        for i, t in enumerate(time[:-1]):
            params['theta0'] = theta0[i]
            driftx[i] = drift(xx[i], params=params)
            xx[i+1] = xx[i] + driftx[i] * dt

            if noise[i] == 0:
                if xx[i, 0] > 0.5:  # stadial conditions
                    if laminar:  # compute the noise for the next laminar period
                        n_laminar = np.ceil(gprnd.rvs(k,
                                                      loc=theta,
                                                      scale=sigma_laminar))
                        laminar = False
                        if i + n_laminar > len(time):
                            n_laminar = len(time) - i
                        noise[i:i+int(n_laminar)] = dt * \
                            obs_laminar - bias_correction

                    else:  # compute the noise for the next turbulent period
                        n_Brownian = np.ceil(mean_xi
                                             + mean_xi*(np.random.random()-0.5))
                        laminar = True
                        if i + n_Brownian > len(time):
                            n_Brownian = len(time) - i
                        noise[i:i+int(n_Brownian)] = (sigma_turbulent
                                                      * np.random.normal(size=int(n_Brownian))
                                                      - obs_laminar) * dt - bias_correction

                else:  # interstadial conditions
                    noise[i] = sigma_interstadial * \
                        np.random.normal() * np.sqrt(dt)

            xx[i+1, 0] -= noise[i]
            xx[i+1, 1] += sigma_theta * np.random.normal() * np.sqrt(dt)

        #########################################################
        # downsampling the trajectory                           #
        #########################################################
        params['theta0'] = 1.3  # is set to 1.3 to store transient trajectories
        directory = dirname(params)
        columns = ['time', 'LR04', 'theta0', 'I', 'theta', 'T', 'S',
                   'dI', 'dtheta', 'dT', 'dS', 'noise']

        #########################################################
        # the data is first smooted with a 5 year running mean  #
        #########################################################

        w = 500
        for idx in range(xx.shape[0]):
            if idx < int(w/2):
                xx[idx] = np.mean(xx[:idx + int(w/2)], axis=0)
                driftx[idx] =  np.mean(driftx[:idx + int(w/2)], axis=0)
                noise[idx] =   np.mean(noise[:idx + int(w/2)], axis=0)
            elif idx > xx.shape[0]-int(w/2):
                xx[idx] = np.mean(xx[idx-int(w/2):], axis=0)
                driftx[idx] = np.mean(driftx[idx-int(w/2):], axis=0)
                noise[idx] =  np.mean(noise[idx-int(w/2):], axis=0)
            else:
                xx[idx] = np.mean(xx[idx-int(w/2):idx + int(w/2)], axis=0)
                driftx[idx] = np.mean(driftx[idx-int(w/2):idx + int(w/2)], axis=0)
                noise[idx] = np.mean(noise[idx-int(w/2):idx + int(w/2)], axis=0)
                
        res1 = 5
        #filtered1 = cheby_lowpass_filter(xx.T, 1/res1, 1/dt, order=4, rp=0.05)
        downsampled1 = xx[::500, :]
        traj5y = pd.DataFrame(np.concatenate((time[:, None][::500],
                                              LR04[:, None][::500],
                                              theta0[:, None][::500],
                                              downsampled1,
                                              driftx[::500,:],
                                              noise[:, None][::500]),
                                             axis=1),
                              columns=columns)
        traj5y.to_csv(directory + '/trans_traj_5y_n%i.csv' % (ntraj),
                      index=False)

        res2 = 20
        #filtered2 = cheby_lowpass_filter(xx.T, 1/res1, 1/dt, order=4, rp=0.05)
        downsampled2 = xx[::2000, :]
        traj20y = pd.DataFrame(np.concatenate((time[:, None][::2000],
                                               LR04[:, None][::2000],
                                               theta0[:, None][::2000],
                                               downsampled2,
                                               driftx[::2000, :],
                                               noise[:,None][::2000]),
                                              axis=1),
                               columns=columns)
        traj20y.to_csv(directory + '/trans_traj_20y_n%i.csv' % (ntraj),
                       index=False)
