import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from os import mkdir
from os.path import isdir, exists
from full_model import gamma_of_ice
from full_model import ocean_atm_jac
from bifurcations_v6_1 import compute_bifurcation_diagram
from miscellanious import dirname
from plot_functions import make_patch_spines_invisible

import tol_colors as tc

colors ={'theta': tc.tol_cset('muted')[5],
         'PIP' : tc.tol_cset('muted')[7],
         'PaTh' : tc.tol_cset('muted')[3],
         'benthic': tc.tol_cset('muted')[4],
         'd18o': tc.tol_cset('muted')[0]}


def plot_atm_ice_bifurcation(params, thetas, recompute = False):

    colormap = plt.cm.get_cmap('Oranges')
    fig = plt.figure(figsize=(80/25.4, 70/25.4))
    #fig, ax = plt.subplots(figsize=(80/25.4, 70/25.4))
    height_ratios = [0.06, 1]
    gs = fig.add_gridspec(2, 1,
                          hspace=0.,
                          wspace=1,
                          left=0.1,
                          right=0.88,
                          bottom=0.09,
                          top=0.95,
                          height_ratios=height_ratios)

    ax = fig.add_subplot(gs[1])
    
    ax.patch.set_visible(False)
    par_theta0 = params['theta0']

    #################################################################
    # plot theta nullclines                                         #
    #################################################################

    for theta0 in thetas:
        params['theta0'] = theta0
        directory = dirname(params)
        if not exists(directory + '/system_fp.csv') or recompute:
            compute_bifurcation_diagram(params, recompute = True)

        atm_ocean_data = pd.read_csv(directory + '/atm_ocean_nullclines.csv')
        system_fp = pd.read_csv(directory + '/system_fp.csv')
        ice_bifurcations = pd.read_csv(directory + '/ice_bifurcations.csv')

        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'strong'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'strong'],
                color=colors['theta'])
        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'weak'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'weak'],
                color=colors['theta'])

        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'unstable'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'unstable'],
                color=colors['theta'], ls=':')
        ax.scatter(system_fp['I'], system_fp['theta'], marker='o',
                   facecolor='r', edgecolor = 'k',
                   zorder=10, label='stable fixed point')
        ax.annotate(r'$\mathbf{S}|_{\theta_0 = %0.2f}$' % (theta0),
                    xy=(system_fp['I'] - 0.3, system_fp['theta'] + 0.05),
                    ha='center', va='center',
                    color='slategray')

    #################################################################
    # plot ice nullclines                                           #
    #################################################################

    ice_data = pd.read_csv(directory + '/ice_nullclines.csv')

    ax.plot(ice_data['I'][ice_data['branch'] == 'ice-rich'],
            ice_data['theta'][ice_data['branch'] == 'ice-rich'],
            color = colors['PIP'])
    ax.plot(ice_data['I'][ice_data['branch'] == 'low-ice'],
            ice_data['theta'][ice_data['branch'] == 'low-ice'],
            color = colors['PIP'])
    ax.plot(ice_data['I'][ice_data['branch'] == 'unstable'],
            ice_data['theta'][ice_data['branch'] == 'unstable'],
            color = colors['PIP'], ls = ':')

    ax.scatter(ice_bifurcations['I'][0],
               ice_bifurcations['theta'][0],
               color = 'k', zorder = 10, s = 15)
    ax.annotate('B2', xy = (ice_bifurcations['I'][0] + 0.2,
                            ice_bifurcations['theta'][0]), 
                ha = 'center', va = 'center')
    ax.scatter(ice_bifurcations['I'][1],
               ice_bifurcations['theta'][1],
               color = 'k', zorder = 10, s = 15, label = 'bifurcation point')
    ax.annotate('B1', xy = (ice_bifurcations['I'][1] - 0.2,
                            ice_bifurcations['theta'][1]),
                ha = 'center', va = 'center')



    #################################################################
    # set labels                                                    #
    #################################################################

    ax.set_ylabel(r'$\theta$', color = colors['theta'])
    ax.set_xlabel('$I$', color = colors['PIP'])
    ax.set_xlim(1.8, -2)
    ax.set_ylim(0.8,1.6)


    ### theta axis ###
    ice_ax = np.arange(2,-2,-0.01)
    rax = ax.twinx()
    rax.set_zorder(-1)
    make_patch_spines_invisible(rax)
    rax.spines['right'].set_visible(True)
    rax.plot(ice_ax, gamma_of_ice(ice_ax, params),
              color='slategray', alpha=1, lw = 0.8)
    rax.set_ylabel(r'$\gamma$', color='slategray')
    rax.set_ylim(0.4,4.1)

    #################################################################
    # plot background                                               #
    #################################################################

    background = fig.add_subplot(gs[0])

    # background = ax.twinx()
    # make_patch_spines_invisible(background)
    background.set_zorder(-3)
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

    # background = ax.twinx()
    # background.set_zorder(-3)
    # #plotlim = plt.xlim() + plt.ylim()  
    # background.imshow([[0,1]], cmap=plt.cm.cool,
    #                   interpolation='bicubic', aspect = 'auto',
    #                   extent = [ax.get_xlim()[0], ax.get_xlim()[1],
    #                             ax.get_ylim()[0], ax.get_ylim()[1]], 
    #                   alpha = 0.2)
    # background.xaxis.set_visible(False)
    # background.yaxis.set_visible(False)
    # background.plot([],[], label = 'stable', color = 'k')
    # background.plot([],[], ls = ':', label = 'unstable', color = 'k')
    # background.legend(frameon = False, loc = 'lower left', fontsize = 6,
    #                   bbox_to_anchor = [0.2, 0.82])

    # ax.annotate('stadial', xy=(0, 1.05), xycoords='axes fraction')
    # ax.annotate('interstadial', xy=(1, 1.05), xycoords='axes fraction', ha = 'right')
    # fig.subplots_adjust(left= 0.16,
    #                     right = 0.85,
    #                     bottom = 0.15,
    #                     top = 0.9)

    params['theta0'] = par_theta0
    directory = dirname(params, sub = 'figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)
    
    fig.savefig(directory + '/figA2.png', dpi = 300)
    fig.savefig(directory + '/figA2.pdf', format = 'pdf')

    plt.close()
