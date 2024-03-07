import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import mkdir
from os.path import isdir, exists
from miscellanious import check_results
from miscellanious import dirname
from stochastic_trajectory import compute_ran_traj
from transition_detection import detect_transitions_sim
from scipy.interpolate import interp1d
from scipy.stats import genpareto as gprnd
from plot_functions import binning, make_patch_spines_invisible
import tol_colors as tc

colors ={'theta': tc.tol_cset('muted')[5],
         'PIP' : tc.tol_cset('muted')[7],
         'PaTh' : tc.tol_cset('muted')[3],
         'benthic': tc.tol_cset('muted')[4],
         'd18o': tc.tol_cset('muted')[0]}


def plot_ran_traj(params, thetas, ntraj=1, recompute=False):

    width = 165.3 / 25.4
    height = 150 / 25.4
    panel = np.array(['(a)', '(b)', '(c)', '(d)','(e)', '(f)', '(g)', '(h)'])

    height_ratios = [1.5, 1,1,1, 2, 1.5, 1,1,1]
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(9,
                          hspace=0,
                          wspace=0.5,
                          left=0.12,
                          right=0.88,
                          bottom=0.1,
                          top=0.95,
                          height_ratios=height_ratios)
    par_theta0 = params['theta0']

    for v, theta0 in enumerate(thetas):
        params['theta0'] = theta0
        filename = 'ran_traj_n%i.csv' % (ntraj)
        traj_name, traj_exists = check_results(params, filename)

        if not traj_exists or recompute:
            print(r'computing ran traj for $\theta = $%0.2f' % (theta0))
            compute_ran_traj(params, theta0, ntraj=ntraj, recompute=True)

        traj = pd.read_csv(traj_name)

        #############################################################
        # Downsampling the data                                     #
        #############################################################

        # KR: data is downsampled before saving

        # Dt = 5
        # t_f = traj['time'].iloc[-1]
        # bins = np.arange(0,t_f+ Dt +1, Dt) - Dt/2
        # binned_time, theta = binning(traj['time'], traj['theta'], bins)
        # binned_time, I = binning(traj['time'], traj['I'], bins)
        # binned_time, T = binning(traj['time'], traj['T'], bins)
        # binned_time, q = binning(traj['time'], traj['T'] - traj['S'], bins)

        axlist = [fig.add_subplot(gs[v * 5 +i]) for i in range(0, 4)]

        #############################################################
        # assessing the transitions                                 #
        #############################################################

        sim_transitions, sim_durations = detect_transitions_sim(traj, params)
        for i, ax in enumerate(axlist):
            make_patch_spines_invisible(ax)
            ax.xaxis.set_visible(False)
            if np.mod(i, 2) == 0:
                ax.spines['left'].set_visible(True)
            else:
                ax.spines['right'].set_visible(True)
                ax.yaxis.set_label_position('right')
                ax.yaxis.set_ticks_position('right')

                
        GI = False
        for start, end in zip(sim_transitions['age'].values[::-1],
                              np.append(sim_transitions['age'].values[-2::-1], 30000)):
            if not GI:
                for ax in axlist:
                    ax.axvspan(start,
                               end,
                               alpha=0.2,
                               edgecolor=None,
                               facecolor='slategray',
                               zorder=1)

            GI = not GI

        axlist[0].set_title(r'$\theta_0 = %0.2f$' % (theta0))
        axlist[0].plot(traj['time'], traj['theta'], label=r'$\theta$', color=colors['theta'],
                       lw=0.6)
        axlist[0].set_ylim(axlist[0].get_ylim()[::-1])
        axlist[0].set_ylabel(r'$\theta$', color=colors['theta'])
        axlist[0].annotate(panel[4*v], (0.02, 0.8), xycoords='axes fraction')

        axlist[1].plot(traj['time'], traj['I'], label='Ice', color=colors['PIP'])
        axlist[1].set_ylim(axlist[1].get_ylim()[::-1])
        axlist[1].set_ylabel('$I$', color=colors['PIP'])
        axlist[1].annotate(panel[1 + 4*v], (0.02, 0.8), xycoords='axes fraction')
        axlist[2].plot(traj['time'], traj['T'] -
                       traj['S'], label='AMOC', color=colors['PaTh'])
        axlist[2].set_ylabel('$q$', color=colors['PaTh'])
        #axlist[2].spines['right'].set_position(('axes', 1.05))
        #axlist[2].yaxis.set_label_coords(1.1, 0.5)
        axlist[2].annotate(panel[2 + 4*v], (0.02, 0.8), xycoords='axes fraction')

        axlist[3].plot(traj['time'], traj['T'], color=colors['benthic'])
        axlist[3].set_ylabel('$T$', color=colors['benthic'])
        #axlist[3].spines['left'].set_position(('axes', -0.02))
        axlist[3].spines['bottom'].set_visible(True)
        axlist[3].set_xlabel('time [y]')
        axlist[3].annotate(panel[3 + 4*v], (0.02, 0.8), xycoords='axes fraction')
        axlist[3].xaxis.set_visible(True)
        axlist[3].set_xticks(np.arange(0, 30001, 5000))
        axlist[3].set_xticks(np.arange(0, 30001, 2500), minor=True)

        for ax in axlist:
            ax.set_xlim(0,30000)

    params['theta0'] = par_theta0
    directory = dirname(params, sub='figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)

    fig.savefig(directory + '/fig05_n%i.png' % (ntraj),
                dpi=300)
    fig.savefig(directory + '/fig05_n%i.pdf' % (ntraj), format='pdf')

    plt.close()
