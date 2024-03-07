import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from roots_v2 import find_roots_1d



#################################################################
# equations of motion                                           #
#################################################################

def gamma_of_ice(Ice, params):
    Delta_gamma = params['Delta_gamma']
    gamma0 = params['gamma0']
    return gamma0 + (np.tanh((-Ice+params['I0'])/params['omega'])+1) * Delta_gamma / 2


def dtheta_dt(theta, T, I, params):
    gamma_eff = gamma_of_ice(I, params)
    return -params['eta'] * (theta - params['theta0']) + gamma_eff*(T-theta)


def dT_dt(theta, T, S, I, params):
    gamma_eff = gamma_of_ice(I, params)
    return -gamma_eff * (T-theta) - (1+params['mu'] * np.abs(T-S)) * T

def dS_dt(T, S, params):
    return params['sigma'] - (params['xi']+params['mu'] * np.abs(T-S)) * S

def dI_dt(I, theta, params):
    albedo = params['Delta']*np.tanh(I/params['ha'])
    I = np.asarray(I)
    mask = I > 0
    export = params['R0'] * mask * I
    net_out_longwave = (-params['L0']
                        + params['L1'] * theta
                        - params['L2'] * I)
    return albedo - export + net_out_longwave

# def dI_dt(I, theta, params):
#     I_fp = (np.tanh((theta - 1)/0.1))
#     return -params['tau_seaice']* (I - I_fp)

def dIdot_dI(I, params):
    albedo = params['Delta']/params['ha'] * (1-np.tanh(I/params['ha']) **2)
    export = params['R0'] * np.heaviside(I, 0)
    olw = params['L2']
    return albedo - export - olw


#################################################################
# equation which is set to zero in order to compute stationary  #
# solutions of the system of coupled ODEs                       #
# this return the stationary q, which yields unique solutions   #
# for theta and T in turn.                                      #
#################################################################

def coupled_fp(q, gamma, params):
    A = 1-gamma/(1+params['mu']*np.abs(q) + gamma)
    B = gamma * params['eta'] * params['theta0'] *\
        (1+params['mu']*np.abs(q)+gamma)
    C = (params['eta'] + gamma) * \
        (1+params['mu']*np.abs(q)+gamma) - gamma**2
    D = (params['xi']+params['mu']*np.abs(q)) * q + params['sigma']
    E = ((1-params['xi']) * gamma * params['eta'] * params['theta0']
         / ((1+params['mu']*np.abs(q)+gamma)*(params['eta']+gamma)-gamma ** 2))
    return A * B/C - D - E


#################################################################
# define functions, which compute theta and T that correspond   #
# to a stationary q                                             #
#################################################################

def T_of_q(q, gamma, params):
    M = gamma * params['eta'] * params['theta0']
    K = (1+gamma + params['mu']*np.abs(q))
    L = (params['eta'] + gamma)
    return M / (K * L - gamma ** 2)


def theta_of_q(q, gamma, params=None):
    A = params['eta'] * params['theta0']
    B = params['eta'] + gamma - \
        gamma ** 2 / (1 + params['mu'] * np.abs(q) + gamma)
    return A / B


def theta_fixed_T(I, T, params):
    gamma_eff = gamma_of_ice(I, params=params)
    return (params['eta'] * params['theta0'] + gamma_eff * T)/(params['eta'] + gamma_eff)


def deterministic_drift(x, params):
    I = x[0]
    theta = x[1]
    T = x[2]
    S = x[3]
    dtheta = dtheta_dt(theta, T, I, params) / params['tau_atm']
    dT = dT_dt(theta, T, S, I, params) / params['tau_ocean']
    dS = dS_dt(T, S, params) / params['tau_ocean']
    dI = dI_dt(I, theta, params) / params['tau_seaice']
    return np.array([dI, dtheta, dT, dS])


def ocean_atm_jac(x, gamma, params):

    theta = x[0]
    T = x[1]
    S = x[2]

    J_stommel = np.zeros((gamma.size, 3, 3))

    J_stommel[:, 0, 0] = -params['eta'] - gamma
    J_stommel[:, 0, 1] = gamma
    J_stommel[:, 0, 2] = 0

    J_stommel[:, 1, 0] = gamma
    J_stommel[:, 1, 1] = (-gamma - 1 - params['mu']*np.abs(T-S)
                          - params['mu'] * T * np.sign(T-S))
    J_stommel[:, 1, 2] = params['mu'] * T * np.sign(T-S)

    J_stommel[:, 2, 0] = 0
    J_stommel[:, 2, 1] = -params['mu'] * S * np.sign(T-S)
    J_stommel[:, 2, 2] = (-1 - params['mu'] * np.abs(T-S)
                          + params['mu'] * S * np.sign(T-S))

    return J_stommel


