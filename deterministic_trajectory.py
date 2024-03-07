import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isdir, exists
from full_model import deterministic_drift
from bifurcations_v6_1 import compute_bifurcation_diagram
from miscellanious import dirname
from miscellanious import check_results


def compute_det_traj(duration, theta0, perturbation, params):

    params['theta0'] = theta0
    filename = 'det_traj_%0.2f.csv' %(perturbation)
    traj_name, traj_exists = check_results(params, filename)

    if traj_exists:
        traj = pd.read_csv(traj_name)
        print('read trajectory from ' + traj_name)

    else: 
        directory = dirname(params)

        if not exists(directory + '/system_fp.csv'):
            compute_bifurcation_diagram(params)
        x0 = pd.read_csv(directory + '/system_fp.csv').values.squeeze()

        # transform the T,q representation into T,S representation
        x0[3] = x0[2] - x0[3]

        dt = 0.01
        time = np.arange(0, duration, dt)
        traj = np.zeros((len(time), 4))
        traj[0] = x0
        print('start Euler integration of det traj')
        for i, t in enumerate(time[:-1]):

            traj[i+1] = traj[i] + deterministic_drift(traj[i], params) * dt

            if (t > 200) & (t < 201):
                traj[i+1, 0] = perturbation

        traj = pd.DataFrame(np.concatenate((time[:,None][::2], traj[::2]),
                                           axis = 1),
                            columns = ['time', 'I', 'theta', 'T', 'S'])
        
        traj.to_csv(traj_name, index = False)
        print('ended Euler integration of det traj')

        
