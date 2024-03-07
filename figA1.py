import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roots_v2 import find_roots_1d
from full_model import coupled_fp
from full_model import theta_of_q
from full_model import T_of_q
from full_model import ocean_atm_jac
from os import mkdir
from os.path import isdir, exists
from plot_functions import make_patch_spines_invisible
from miscellanious import dirname
import tol_colors as tc


colors ={'theta': tc.tol_cset('muted')[5],
         'PIP' : tc.tol_cset('muted')[7],
         'PaTh' : tc.tol_cset('muted')[3],
         'benthic': tc.tol_cset('muted')[4],
         'd18o': tc.tol_cset('muted')[0]}

#################################################################
# setup                                                         #
#################################################################

def plot_atm_ocean_bifurcation(params):

    # LOAD DATA

    directory = dirname(params)
    atm_ocean_data = pd.read_csv(directory + '/atm_ocean_nullclines.csv')

    fig = plt.figure(figsize=(80/25.4, 90/25.4))
    height_ratios = [0.6] + [1] * 7
    fig = plt.figure(figsize=(80/25.4, 90/25.4))
    gs = fig.add_gridspec(8, 1,
                          hspace=0.5,
                          left=0.18,
                          bottom=0.2,
                          top=0.9,
                          right=0.8,
                          height_ratios = height_ratios)
    #                      width_ratios = width_ratios,
    #                      height_ratios = height_ratios)

    theta_ax = fig.add_subplot(gs[1:4])
    q_ax = fig.add_subplot(gs[3:6])
    T_ax = fig.add_subplot(gs[5:8])
    background = fig.add_subplot(gs[0])

    make_patch_spines_invisible(theta_ax)
    make_patch_spines_invisible(q_ax)
    make_patch_spines_invisible(T_ax)
    background.xaxis.set_visible(False)
    background.yaxis.set_visible(False)

    xlim = (0.5, 2)

    ### theta axis ###
    theta_ax.spines['left'].set_visible(True)
    theta_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'strong'],
                  atm_ocean_data['theta'][atm_ocean_data['branch'] == 'strong'],
                  color=colors['theta'])
    theta_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'weak'],
                  atm_ocean_data['theta'][atm_ocean_data['branch'] == 'weak'],
                  color=colors['theta'])

    theta_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'unstable'],
                  atm_ocean_data['theta'][atm_ocean_data['branch']
                                          == 'unstable'],
                  color=colors['theta'], ls = ':')

    theta_ax.set_ylabel(r'$\theta$', color = colors['theta'])
    theta_ax.spines['left'].set_color(colors['theta'])
    theta_ax.tick_params(axis = 'y', colors = colors['theta'])
    theta_ax.set_xlim(xlim)
    theta_ax.set_ylim((0.98,1.22))
    theta_ax.xaxis.set_visible(False)
    # theta_ax.annotate('stadial', xy=(0, 1.1), xycoords='axes fraction', ha = 'left')
    # theta_ax.annotate('interstadial', xy=(1, 1.1), xycoords='axes fraction', ha = 'right')

    # theta_ax.annotate('strong mode', xy=(3, 1.25),  xycoords='data',
    #                   xytext=(3,1.15 ), textcoords='data',
    #                   arrowprops=dict(facecolor='black', 
    #                                   width = 0.8,
    #                                   headwidth = 3,
    #                                   headlength = 4,
    #                                   shrink = 1),
    #                   horizontalalignment='center', verticalalignment='center')

    ### q axis ###
    q_ax.spines['right'].set_visible(True)
    q_ax.yaxis.set_ticks_position('right')
    q_ax.yaxis.set_label_position('right')
    q_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'strong'],
              atm_ocean_data['q'][atm_ocean_data['branch'] == 'strong'],
              color = colors['PaTh'])
    q_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'weak'],
              atm_ocean_data['q'][atm_ocean_data['branch'] == 'weak'],
              color = colors['PaTh'])
    q_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'unstable'],
              atm_ocean_data['q'][atm_ocean_data['branch'] == 'unstable'],
              color = colors['PaTh'], ls = ':')
    q_ax.set_ylabel(r'$q$', color = colors['PaTh'])
    q_ax.spines['right'].set_color(colors['PaTh'])
    q_ax.tick_params(axis = 'y', colors = colors['PaTh'])
    q_ax.xaxis.set_visible(False)
    q_ax.set_xlim(xlim)

    ### T axis ###
    T_ax.spines['left'].set_visible(True)
    T_ax.spines['bottom'].set_visible(True)
    T_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'strong'],
              atm_ocean_data['T'][atm_ocean_data['branch'] == 'strong'],
              color = colors['benthic'])
    T_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'weak'],
              atm_ocean_data['T'][atm_ocean_data['branch'] == 'weak'],
              color = colors['benthic'])
    T_ax.plot(atm_ocean_data['gamma'][atm_ocean_data['branch'] == 'unstable'],
              atm_ocean_data['T'][atm_ocean_data['branch'] == 'unstable'],
              color = colors['benthic'], ls = ':')
    T_ax.set_ylabel(r'$T$', color =colors['benthic'])
    T_ax.spines['left'].set_color(colors['benthic'])
    T_ax.set_ylim(0.2, 0.7)
    T_ax.set_xlim(xlim)
    T_ax.spines['right'].set_color(colors['benthic'])
    T_ax.tick_params(axis = 'y', colors = colors['benthic'])
    T_ax.set_xlabel(r'$\gamma$')

    background.set_zorder(-2)
    #plotlim = plt.xlim() + plt.ylim()

    background.imshow([[0, 1]], cmap=tc.tol_cmap('BuRd'),
                      interpolation='bicubic', aspect='auto',
                      extent=[background.get_xlim()[0], background.get_xlim()[1],
                              background.get_ylim()[0], background.get_ylim()[1]],
                      alpha=1)
    background.xaxis.set_visible(False)
    background.yaxis.set_visible(False)
    background.annotate('stadial', xy=(0.02, 0.45),
                        xycoords='axes fraction',
                        va = 'center',
                        color = 'whitesmoke',
                        fontsize = 8)

    background.annotate('interstadial', xy=(0.98, 0.45),
                        xycoords='axes fraction',
                        ha='right',
                        va = 'center',
                        color = 'k',
                        fontsize = 8)

    # background.imshow([[0,0.6, 1]], cmap=plt.cm.cool,
    #                   interpolation='bicubic', aspect = 'auto',
    #                   alpha = 0.2)
    #           #extent=plotlim)  
    # background.xaxis.set_visible(False)
    # background.yaxis.set_visible(False)
    # background.plot([],[], label = 'stable', color = 'k')
    # background.plot([],[], ls = ':', label = 'unstable', color = 'k')
    # background.legend(frameon = False)

    directory = dirname(params, sub = 'figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)

    fig.savefig(directory + '/figA1.png', dpi = 300)
    fig.savefig(directory + '/figA1.pdf')

    plt.close()

    #################################################################
    # test the supposedly unstable solutions of the stable branch   #
    #################################################################

    # gamma = q_list[:,0][unstable][10]
    # q = q_list[:,1][unstable][10]
    # T = T_list[:,1][unstable][10]
    # S = T-q
    # theta = theta_list[:,1][unstable][10]

    # test = ocean_atm_jac(np.array([theta, T, S]), gamma)
    # w, v = np.linalg.eig(test)

    # ocean_atm(np.array([theta,T, S]), gamma = gamma)

    # perturbation = np.random.normal(scale = 0.1, size = 3)

    # dt = 0.001
    # time = np.arange(0,100,dt)
    # xx = np.zeros((len(time), 3))
    # xx[0] = np.array([theta, T, S]) + perturbation

    # for i in range(len(time)-1):
    #     xx[i+1] = xx[i] + ocean_atm(xx[i], gamma = gamma)* dt

    # fig, ax = plt.subplots()
    # ax.plot(time, xx[:, 0])
    # ax.plot(time, xx[:, 1])
    # ax.plot(time, xx[:, 2])
