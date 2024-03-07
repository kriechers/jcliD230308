import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir, exists
from miscellanious import dirname
from miscellanious import check_results
from plot_functions import make_patch_spines_invisible
from deterministic_trajectory import compute_det_traj
from bifurcations_v6_1 import compute_bifurcation_diagram
from full_model import gamma_of_ice
from scipy.optimize import fsolve
import tol_colors as tc
import matplotlib.colors as mcolors
import colorsys


colors ={'theta': tc.tol_cset('muted')[5],
         'PIP' : tc.tol_cset('muted')[7],
         'PaTh' : tc.tol_cset('muted')[3],
         'benthic': tc.tol_cset('muted')[4],
         'd18o': tc.tol_cset('muted')[0]}



def plot_det_trajs(params, perturbations, thetas, recompute = False):


    d18o_map = [mcolors.hex2color(colors['theta'])]
    PIP_map = [mcolors.hex2color(colors['PIP'])]
    PaTh_map =[mcolors.hex2color(colors['PaTh'])]
    benthic_map = [mcolors.hex2color(colors['benthic'])]

    d18o = colorsys.rgb_to_hsv(*d18o_map[0])
    PIP =  colorsys.rgb_to_hsv(*PIP_map[0])
    PaTh = colorsys.rgb_to_hsv(*PaTh_map[0])
    benthic = colorsys.rgb_to_hsv(*benthic_map[0])

        
    for i in range(len(perturbations)-1):
        #d18o_v = (1 - d18o[2]) * (i+1) /len(perturbations) + d18o[2]
        d18o_v =  (1 - PIP[2]) * (i+1) /len(perturbations) + PIP[2]
        PIP_v = (1 - PIP[2]) * (i+1) /len(perturbations) + PIP[2]
        PaTh_v =  (1 - PaTh[2]) * (i+1) /len(perturbations) + PaTh[2]
        benthic_v = benthic[2]  * (len(perturbations) - i-1)/len(perturbations)
        d18o_map.append(colorsys.hsv_to_rgb(d18o[0],
                                            d18o[1],
                                            d18o_v))
        PIP_map.append(colorsys.hsv_to_rgb(PIP[0],
                                           PIP[1],
                                           PIP_v))
        PaTh_map.append(colorsys.hsv_to_rgb(PaTh[0],
                                            PaTh[1],
                                            PaTh_v))
        benthic_map.append(colorsys.hsv_to_rgb(benthic[0],
                                               benthic[1],
                                               benthic_v))

        
    colormaps = {'theta': plt.cm.get_cmap('Purples'),
                 'I': tc.tol_cmap('YlOrBr'),
                 'q': plt.cm.get_cmap('Greens'),
                 'T': plt.cm.get_cmap('Blues')}

    par_theta0 = params['theta0']
    lw = 1.5
    lwt = 1.2

    width = 165.3 / 25.4
    height = 180 / 25.4

    theta0 = thetas
    labels = np.array([['(a)', '(f)'],
                       ['(b)', '(g)'],
                       ['(c)', '(h)'],
                       ['(d)', '(i)'],
                       ['(e)', '(j)']])

    height_ratios = [1.2] + [1.2] * 9 + [5] + [0.5] * 3 + [1] * 16
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(30, 4,
                          hspace=0.3,
                          wspace=1,
                          left=0.1,
                          right=0.88,
                          bottom=0.08,
                          top=0.94,
                          height_ratios=height_ratios)

    n = len(perturbations) + 2
    durations = np.zeros((2, len(perturbations)))

    for v, theta0 in enumerate(thetas):

        params['theta0'] = theta0

        axlist = [fig.add_subplot(gs[i:i+4, 2*v:2*(v+1)])
                  for i in range(14, 30, 4)]
        if v == 0:
            ax = fig.add_subplot(gs[3:10, :2])
            ax_qt = fig.add_subplot(gs[10, :2])  # axis for q and T
            rax_qt = ax_qt.twinx()
            ax_qt.set_ylabel(r'$q$', color=colors['PaTh'], labelpad=8)
        else:
            ax = fig.add_subplot(gs[3:10, 2:])
            ax_qt = fig.add_subplot(gs[10, 2:])  # axis for q and T
            rax_qt = ax_qt.twinx()
            rax_qt.set_ylabel(r'${T}$', color=colors['benthic'])

        #################################################################
        # plot the trajectory in the state space                        #
        #################################################################

        directory = dirname(params)
        if not exists(directory + '/system_fp.csv'):
            compute_bifurcation_diagram(params)

        atm_ocean_data = pd.read_csv(directory + '/atm_ocean_nullclines.csv')
        system_fp = pd.read_csv(directory + '/system_fp.csv')
        ice_bifurcations = pd.read_csv(directory + '/ice_bifurcations.csv')
        ice_data = pd.read_csv(directory + '/ice_nullclines.csv')
        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'strong'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'strong'],
                color=colors['theta'], lw = lw, label = r'$\theta$-nullcline')
        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'weak'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'weak'],
                color=colors['theta'], lw = lw)
        ax.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'unstable'],
                atm_ocean_data['theta'][atm_ocean_data['branch'] == 'unstable'],
                color=colors['theta'], ls=':',
                lw = lw)


        ax.axvline(system_fp['I'][0], color = 'gray', lw = 0.6)


        ax.plot(ice_data['I'][ice_data['branch'] == 'ice-rich'],
                ice_data['theta'][ice_data['branch'] == 'ice-rich'],
                color=colors['PIP'], lw = lw, label = r'$I$-nullcline')
        ax.plot(ice_data['I'][ice_data['branch'] == 'low-ice'],
                ice_data['theta'][ice_data['branch'] == 'low-ice'],
                color=colors['PIP'], lw = lw)
        ax.plot(ice_data['I'][ice_data['branch'] == 'unstable'],
                ice_data['theta'][ice_data['branch'] == 'unstable'],
                color=colors['PIP'], ls=':', lw = lw)
        ice_axis = np.arange(-2, 2, 0.01)
        theta_fixed_T = ((params['eta'] * params['theta0']
                          + gamma_of_ice(ice_axis, params) * system_fp['T'].values)
                         / (params['eta'] + gamma_of_ice(ice_axis, params)))

        ax.plot(ice_axis, theta_fixed_T,
                color=colors['theta'],
                alpha=0.5, ls = '--', lw = lw)

        ax.scatter(system_fp['I'][0], system_fp['theta'], marker='o',
                   facecolor='red', edgecolor = 'k',
                   zorder=10)
                   #label = 'stable fixed point')


        # ax.annotate('stadial', xy=(0, 1.2), xycoords='axes fraction')
        # ax.annotate('interstadial', xy=(1, 1.2),
        #             xycoords='axes fraction', ha='right')

        ax.yaxis.label.set_rotation(90)
        ax.yaxis.set_label_coords(-0.2, 0.5)
        ax.set_xlim(1.8, -2.2)
        #ax.set_ylim(0.8, theta0 + 0.1)
        ax.set_ylim(0.7, 1.7)
        ax.set_xlabel(r'${I}$', color=colors['PIP'])
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')

        ax.annotate(labels[0, v], (0.07, 0.1), xycoords='axes fraction')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.patch.set_visible(False)
        ice_bifurcations = pd.read_csv(directory + '/ice_bifurcations.csv')
        #if v == 1:

        ax.scatter(ice_bifurcations['I'][0],
                   ice_bifurcations['theta'][0],
                   color='k', zorder=10, s=15)

        ax.scatter(ice_bifurcations['I'][1],
                   ice_bifurcations['theta'][1],
                   color='k', zorder=10, s=15)

        ax.annotate('B1', xy=(ice_bifurcations['I'][1] - 0.3,
                              ice_bifurcations['theta'][1]),
                    ha='center', va='center',
                    fontsize = 6)
        ax.annotate('B2', xy=(ice_bifurcations['I'][0] + 0.3,
                              ice_bifurcations['theta'][0]),
                    ha='center', va='center',
                    fontsize = 6)

        tax = ax.twiny()
        make_patch_spines_invisible(tax)
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(perturbations)
        tax.spines['top'].set_position(('axes', 0.45 + 0.3*v))
        tax.tick_params(axis='x', labelrotation=270, labelsize=6)
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks_position('top')

        #################################################################
        # plot background                                               #
        #################################################################

        background = fig.add_subplot(gs[0, 2*v: 2*v+2])

        # background = ax.twinx()
        # make_patch_spines_invisible(background)
        background.set_zorder(-3)
        #plotlim = plt.xlim() + plt.ylim()
        background.imshow([[0, 1]], cmap=tc.tol_cmap('BuRd'),
                          interpolation='bicubic', aspect='auto',
                          extent=[ax.get_xlim()[0], ax.get_xlim()[1],
                                  ax.get_ylim()[0], ax.get_ylim()[1]],
                          alpha=1)
        background.xaxis.set_visible(False)
        background.yaxis.set_visible(False)
        background.annotate(r'\textbf{stadial}', xy=(0.02, 0.4),
                            xycoords='axes fraction',
                            va = 'center',
                            color = 'whitesmoke')
        
        background.annotate(r'\textbf{interstadial}', xy=(0.98, 0.4),
                            xycoords='axes fraction',
                            ha='right',
                            va = 'center',
                            color = 'k')
        

        # background.plot([],[], label = 'stable', color = 'k')
        # background.plot([],[], ls = ':', label = 'unstable', color = 'k')
        # background.legend(frameon = False, loc = 'lower left', fontsize = 6,
        #                   bbox_to_anchor = [0.2, 0.82])

        #################################################################
        # plot legend                                                   #
        #################################################################

        ax.plot([],[], color = colors['PaTh'], label = '$q$-nullcline')
        ax.plot([],[], color = colors['benthic'], label = '$T$-nullcline')

        if v ==0: 
            legend = ax.legend(ncols = 2,
                               loc = 'upper left',
                               bbox_to_anchor = (0.23,0.98),
                               fontsize = 7,
                               handlelength = 1,
                               frameon = False)

        xdata = legend.legendHandles[2].get_xdata()
        #legend.legendHandles[2].set_xdata(xdata - 10)
        #legend.texts[2].set_x(-10)
        xdata = legend.legendHandles[3].get_xdata()
        #legend.legendHandles[3].set_xdata(xdata - 10)
        #legend.texts[3].set_x(-10)


        #################################################################
        # plot q and T                                                  #
        #################################################################

        rax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'strong'],
                    atm_ocean_data['T'][atm_ocean_data['branch'] == 'strong'],
                    color=colors['benthic'], lw = lw)
        rax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'weak'],
                    atm_ocean_data['T'][atm_ocean_data['branch'] == 'weak'],
                    color=colors['benthic'], lw = lw)
        rax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'unstable'],
                    atm_ocean_data['T'][atm_ocean_data['branch'] == 'unstable'],
                    color=colors['benthic'], ls=':', lw = lw)

        ax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'strong'],
                   atm_ocean_data['q'][atm_ocean_data['branch'] == 'strong'],
                   color=colors['PaTh'], lw = lw)
        ax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'weak'],
                   atm_ocean_data['q'][atm_ocean_data['branch'] == 'weak'],
                   color=colors['PaTh'], lw = lw)
        ax_qt.plot(atm_ocean_data['I'][atm_ocean_data['branch'] == 'unstable'],
                   atm_ocean_data['q'][atm_ocean_data['branch'] == 'unstable'],
                   color=colors['PaTh'], ls=':', lw = lw)

        ax_qt.set_xlim(ax.get_xlim())
        # xlabels = [0.6, 0.8, 1.5, 3, 3.7, 3.9]
        # xticks = np.zeros_like(xlabels)
        # for i, xl in enumerate(xlabels):
        #     def f(x): return gamma_of_ice(x, params) - xl
        #     x0 = ice_axis[np.sum(f(ice_axis) > 0)]
        #     temp = fsolve(f, x0)
        #     #print(xl, gamma_of_ice(temp, params))
        #     xticks[i] = temp
        # ax_qt.set_xticks(xticks)
        # ax_qt.set_xticklabels(xlabels)
        xticks = np.round(gamma_of_ice(ax.get_xticks(),
                                       params), 1)
        ax_qt.set_xticklabels(xticks)

        ax_qt.set_yticks([0, 0.2])
        rax_qt.set_yticks([0.4, 0.6])
        ax_qt.set_xlabel(r'${\gamma}$')
        ax_qt.axvline(system_fp['I'][0], color = 'gray', lw = 0.6)
        rax_qt.set_ylim(0.2,0.8)
        ax_qt.set_ylim(-0.15,0.4)

        ax_qt.patch.set_visible(False)
        rax_qt.patch.set_visible(False)

        rax_qt.scatter(system_fp['I'][0], system_fp['T'], marker='o',
                      facecolor='red', edgecolor = 'k', 
                      zorder=10)
        ax_qt.scatter(system_fp['I'][0], system_fp['q'], marker='o',
                      facecolor='red', edgecolor = 'k', 
                      zorder=10)

        # background = ax_qt.twinx()
        # make_patch_spines_invisible(background)
        # background.set_zorder(-3)
        # #plotlim = plt.xlim() + plt.ylim()
        # background.imshow([[0, 1]], cmap=plt.cm.cool,
        #                   interpolation='bicubic', aspect='auto',
        #                   extent=[ax_qt.get_xlim()[0], ax_qt.get_xlim()[1],
        #                           ax_qt.get_ylim()[0], ax_qt.get_ylim()[1]],
        #                   alpha=0.2)
        # background.xaxis.set_visible(False)
        # background.yaxis.set_visible(False)

        ##########################################################
        # plotting the trajectories against time in the 4 dims   #
        ##########################################################
        

        for i, axis in enumerate(axlist):
            make_patch_spines_invisible(axis)
            axis.xaxis.set_visible(False)
            axis.annotate(labels[i+1, v], (0.9, 0.85), xycoords='axes fraction')

            if np.mod(i, 2) == 0:
                axis.spines['left'].set_visible(True)
            else:
                axis.spines['right'].set_visible(True)
                axis.yaxis.set_label_position('right')
                axis.yaxis.set_ticks_position('right')

        for j, p in enumerate(perturbations):
            filename = 'det_traj_%0.2f.csv' % (p)
            traj_name, traj_exists = check_results(params, filename)

            if not traj_exists or recompute:
                print(
                    r'computing det traj for $\theta = $%0.2f and $p$ = %0.1f' % (theta0, p))
                compute_det_traj(12000, theta0, p, params)

            det_traj = pd.read_csv(traj_name)

            #########################################################
            # compute duration of interstadial                      #
            #########################################################

            #duration = np.sum(xx[:,3] < system_fp['Ice'].values-0.1) * dt
            time = det_traj['time'].values
            dt = time[1]-time[0]
            duration = np.sum(det_traj['I'] < 0.5) * dt
            durations[v, j] = duration

            axlist[0].plot(time, det_traj['theta'].values,
                           color=d18o_map[::-1][j],
                           lw=lwt,
                           zorder=n-j,
                           label = p)

            axlist[1].plot(time, det_traj['I'], label='Ice',
                           color=PIP_map[::-1][j],
                           lw=lwt,
                           zorder=n-j)

            axlist[2].plot(time, det_traj['T'].values - det_traj['S'].values,
                           color=PaTh_map[::-1][j],
                           lw=lwt,
                           zorder=n-j)

            axlist[3].plot(time, det_traj['T'],
                           color=benthic_map[j],
                           lw=lwt,
                           zorder=n-j)

            ax.plot(det_traj['I'].values, det_traj['theta'].values,
                    color=plt.cm.get_cmap('Greys')((j+2)/n),
                    lw=lwt,
                    zorder=n-j)

        ax.set_title(r'$\theta_{0} = %0.2f$' % (theta0), pad=60, loc='center')

        Ic = ice_bifurcations['I'][0]
        axlist[1].axhline(Ic,
                          ls='--', lw=0.8, color='r')
        # axlist[1].axhline(0,
        #                  ls = '--', lw = 0.8, color ='k')

        axlist[0].set_ylim(axlist[0].get_ylim()[::-1])
        if v == 0:
            ax.set_ylabel(r'${\theta}$', color=colors['theta'])
            axlist[0].set_ylabel(r'$\theta$', color=colors['theta'])
            axlist[2].set_ylabel(r'${q}$', color=colors['PaTh'])

        else:
            axlist[1].set_ylabel(r'${I}$', color=colors['PIP'])
            axlist[3].set_ylabel(r'${T}$', color=colors['benthic'])


        # axlist[0].set_ylim(1.5, 0.8)
        # axlist[2].set_ylim(-0.1, 0.3)
        # axlist[3].set_ylim(0.3, 0.7)
        
        axlist[1].set_ylim(axlist[1].get_ylim()[::-1])
        axlist[1].set_yticks([system_fp['I'].values[0], Ic, -1.5])
        ticklabels = [r'$I_{\mathrm{s}} = %0.2f$' % (system_fp['I'].values[0]),
                      r'$I_{\mathrm{B2}} = %0.2f$' % (Ic),
                      -1.5]
        axlist[1].set_yticklabels(ticklabels)
        temp = axlist[1].get_yticklabels()
        temp[0].set_fontsize(8)
        temp[1].set_fontsize(8)

        axlist[1].yaxis.set_label_coords(1.15, 0.63)


        axlist[3].spines['bottom'].set_visible(True)
        axlist[3].set_xlabel('time [y]')
        axlist[3].xaxis.set_visible(True)

        if v == 1:
            for axis in axlist:
                axis.set_xlim((0, 4000))
            # axlist[0].legend(frameon = False,
            #                  bbox_to_anchor = (0.9,0.98),
            #                  loc = 'upper right',
            #                  ncols = 2,
            #                  handlelength = 1,
            #                  title = r'$I_{\mathrm{p}}$',
            #                  fontsize = 6)

        if v == 0:
            for axis in axlist:
                axis.set_xlim((0, 10000))

    params['theta0'] = par_theta0
    directory = dirname(params, sub='figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)

    fig.savefig(directory + '/fig06.png',
                dpi=300)
    fig.savefig(directory + '/fig06.pdf', format='pdf')

    plt.close()
