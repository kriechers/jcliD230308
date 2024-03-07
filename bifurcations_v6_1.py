import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir, exists
from scipy.interpolate import interp1d
from miscellanious import check_results
from miscellanious import dirname
from full_model import gamma_of_ice
from full_model import coupled_fp
from full_model import T_of_q
from full_model import theta_of_q
from full_model import dI_dt
from full_model import dIdot_dI
from full_model import ocean_atm_jac
from roots_v2 import find_roots_1d


def compute_bifurcation_diagram(params, recompute=False):

    atm_ocean_file, atm_ocean_exists = check_results(params,
                                                     'atm_ocean_nullclines.csv')

    directory = dirname(params)
    if not isdir(directory):
        mkdir(directory)

    if atm_ocean_exists:
        print('reading results from ' + atm_ocean_file)
        atmo_stability = pd.read_csv(atm_ocean_file)

    if not atm_ocean_exists or recompute:

        I_axis = np.arange(-2, 2, 0.005)
        gamma_axis = gamma_of_ice(I_axis, params)
        theta_axis = np.arange(0, 4, 0.01)

        q_fp = []
        theta_fp = []
        T_fp = []
        atm_ocean_stability = []

        #############################################################
        # computing the nullcline of the ocean-atmosphere model wrt #
        # the sea ice                                               #
        #############################################################

        for I, gamma in zip(I_axis, gamma_axis):
            temp = find_roots_1d(coupled_fp, -0.8, 0.8,
                                 gamma,
                                 params,
                                 gp=2000)

            if temp.size == 0:
                continue

            for q in temp:
                theta = theta_of_q(q, gamma, params)
                T = T_of_q(q, gamma, params)
                q_fp.append((I, q))
                theta_fp.append((I, theta))
                T_fp.append((I, T))

                S = T - q
                w, _ = np.linalg.eig(ocean_atm_jac(np.array([theta, T, S]),
                                                   gamma,
                                                   params))

                if all((w < 0).flatten()):
                    stability = True
                else:
                    stability = False

                atm_ocean_stability.append(stability)

        #############################################################
        # assessing stability of atm - ocean nullclines and storing #
        # the results                                               #
        #############################################################

        theta_fp_list = np.array([theta_fp]).squeeze()
        q_fp_list = np.array([q_fp]).squeeze()
        T_fp_list = np.array([T_fp]).squeeze()
        atm_ocean_stability_list = np.array([atm_ocean_stability]).squeeze()
        stable1 = (atm_ocean_stability_list) & (q_fp_list[:, 1] < 0)
        stable2 = (atm_ocean_stability_list) & (q_fp_list[:, 1] > 0)
        unstable = ~atm_ocean_stability_list

        variables = ['gamma', 'I', 'theta', 'q', 'T']
        data = np.concatenate((gamma_of_ice(theta_fp_list[:, 0], params)[:, None],
                               theta_fp_list[:, :],
                               q_fp_list[:, 1][:, None],
                               T_fp_list[:, 1][:, None]),
                              axis=1)
        atmo_stability = pd.DataFrame(data,  columns=variables)
        atmo_stability['branch'] = 'weak'
        atmo_stability.loc[np.argwhere(
            unstable).flatten(), 'branch'] = 'unstable'
        atmo_stability.loc[np.argwhere(stable2).flatten(), 'branch'] = 'strong'
        print('saving atmosphere-ocean bifurcation diagram to ' + atm_ocean_file)
        atmo_stability.to_csv(atm_ocean_file, index=False)

    #############################################################
    # computing the nullcline of the Lohmann sea ice model      #
    #############################################################

    ice_file, ice_exists = check_results(params,
                                         'ice_nullclines.csv')

    if ice_exists:
        print('reading results from ' + ice_file)
        ice_stability = pd.read_csv(ice_file)

    if not ice_exists or recompute:
        I_fp = []
        I_stability = []
        for theta in theta_axis:
            temp = find_roots_1d(dI_dt, -2, 2,
                                 theta,
                                 params,
                                 gp=4000)

            if temp.size == 0:
                continue

            for I in temp:
                I_fp.append((I, theta))
                I_stability.append(dIdot_dI(I, params) < 0)

        #############################################################
        # assessing stability of sea ice nullclines and storing     #
        # the results                                               #
        #############################################################

        I_fp_list = np.array(I_fp).squeeze()
        J = dIdot_dI(I_fp_list[:, 0], params)
        ice_stable = J < 0
        ice_cover = I_fp_list[:, 0] > 0
        mask1 = ice_stable * ice_cover
        mask2 = ice_stable * ~ice_cover
        mask3 = ~ice_stable

        variables = ['I', 'theta']
        ice_stability = pd.DataFrame(I_fp_list,  columns=variables)
        ice_stability['branch'] = 'low-ice'
        ice_stability.loc[np.argwhere(mask3).flatten(), 'branch'] = 'unstable'
        ice_stability.loc[np.argwhere(mask1).flatten(), 'branch'] = 'ice-rich'
        print('saving ice bifurcation diagram to ' + ice_file)
        ice_stability.to_csv(ice_file, index=False)

    #############################################################
    # compute the sea ice bifurcation points                    #
    #############################################################

    ice_bifurcations_file, ice_bifurcations_exsits = check_results(
        params, 'ice_bifurcations.csv')

    I_b = find_roots_1d(dIdot_dI, -1, 2, params,  gp=2000)
    theta_b = [(-params['Delta']*np.tanh(I_b[0]/params['ha'])
                + params['R0'] * np.heaviside(I_b[0], 0)*I_b[0]
                + params['L0']
                + params['L2'] * I_b[0])/params['L1'],
               (-params['Delta']*np.tanh(I_b[1]/params['ha'])
                + params['R0'] * np.heaviside(I_b[1], 0)*I_b[1]
                + params['L0']
                + params['L2'] * I_b[1])/params['L1']]

    print('the bifurcation points of the sea ice are given by'
          + '(I_b = %0.2f, theta_b = %0.2f) and ' % (I_b[0], theta_b[0])
          + ' (I_b = %0.2f, theta_b = %0.2f)' % (I_b[1], theta_b[1]))

    bifurcation_points = pd.DataFrame(np.array([[I_b[0], theta_b[0]],
                                                [I_b[1], theta_b[1]]]),
                                      columns=['I', 'theta'])
    bifurcation_points.to_csv(ice_bifurcations_file, index=False)

    #################################################################
    # computing fixed points of the coupled model                   #
    #################################################################

    system_fp_file, system_fp_exists = check_results(params,
                                                     'system_fp.csv')

    if system_fp_exists:
        system_fp = pd.read_csv(system_fp_file)
        print('reading results from ' + system_fp_file)

    if not system_fp_exists or recompute:

        Ice_nullcline = interp1d(*I_fp_list.T)
        theta_nullcline = interp1d(*theta_fp_list.T)
        T_nullcline = interp1d(*T_fp_list.T)
        q_nullcline = interp1d(*q_fp_list.T)

        x_min = np.maximum(np.min(Ice_nullcline.x), np.min(theta_nullcline.x))
        x_max = np.minimum(np.max(Ice_nullcline.x), np.max(theta_nullcline.x))
        system_fp = pd.DataFrame()
        system_fp['I'] = find_roots_1d(lambda x: Ice_nullcline(x)
                                       - theta_nullcline(x), x_min+0.01, x_max)
        system_fp['theta'] = theta_nullcline(system_fp['I'].values)
        system_fp['T'] = T_nullcline(system_fp['I'].values)
        system_fp['q'] = q_nullcline(system_fp['I'].values)
        system_fp.to_csv(system_fp_file, index=False)

    print('the systemwide fp is')
    print(system_fp)


