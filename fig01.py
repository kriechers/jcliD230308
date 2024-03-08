import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from plot_functions import binning
import pandas as pd
from os import mkdir
from os.path import isdir, exists
from scipy.interpolate import interp1d
from transition_detection import detect_transitions_sim
from transition_detection import transitions_proxy_record
from miscellanious import dirname
# from plot_functions import binning
# from plot_functions import cheby_lowpass_filter
from plot_functions import make_patch_spines_invisible
# from plot_functions import vmarker
from plot_functions import NGRIP_stadial_mask
import tol_colors as tc


def create_fig01(params, ntraj=0):

    #################################################################
    # import simulation data                                        #
    #################################################################

    directory = dirname(params)
    sim = pd.read_csv(directory + '/trans_traj_20y_n%i.csv' % (ntraj))

    # ----------------------------------------------------------------

    #################################################################
    # detect transition in simulation                               #
    #################################################################

    sim_transitions, sim_durations = detect_transitions_sim(sim, params)
    prox_transitions, prox_durations = transitions_proxy_record()

    #################################################################
    # Import NGRIP data                                             #
    #################################################################

    d18o_file = 'proxy_data/GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_Seierstad_et_al._2014_version_10Dec2014-2.xlsx'

    data = pd.read_excel(d18o_file,
                         sheet_name='3) d18O and Ca 20 yrs mean',
                         header=None,
                         skiprows=range(52),
                         names=['age', 'd18o'],
                         usecols='A,E')

    data.drop_duplicates(subset='age', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.dropna(inplace=True)
    d18o = data['d18o'].values
    d18o_age = data['age'].values

    # ----------------------------------------------------------------

    #################################################################
    # import Henry et al. Pa/Th                                     #
    #################################################################

    PaTh_file = 'proxy_data/henry2016.xls'

    PaTh_data = pd.read_excel(PaTh_file,
                              sheet_name='PaTh',
                              header=None,
                              skiprows=range(3),
                              names=['age', 'PaTh', 'Error'],
                              usecols='B,C, D')
    PaTh_age = PaTh_data['age'].values * 1000
    PaTh = PaTh_data['PaTh'].values

    # ----------------------------------------------------------------

    #################################################################
    # import Sadatzki et al. PIP                                    #
    #################################################################

    PIP_file = 'proxy_data/Sadatzki2020/pnas.2005849117.sd02.xlsx'

    PIP_data = pd.read_excel(PIP_file,
                             sheet_name='TOC, biomarkers - MD95-2010',
                             header=None,
                             skiprows=range(1),
                             names=['age', 'PIP'],
                             usecols='H,P')

    PIP_age = PIP_data['age'].values * 1000
    PIP = PIP_data['PIP'].values

    # ----------------------------------------------------------------

    #################################################################
    # import Sadatzki benthic d18o (deep water temperature)         #
    #################################################################

    benthic_file = 'proxy_data/Sadatzki2020/pnas.2005849117.sd06.xlsx'

    benthic_data = pd.read_excel(benthic_file,
                                 sheet_name='ARM, SST, be d18O - MD99-2284',
                                 header=None,
                                 skiprows=range(1),
                                 names=['age', 'benthic'],
                                 usecols='N,O')

    benthic_age = benthic_data['age'].values * 1000
    benthic = benthic_data['benthic'].values

    # ----------------------------------------------------------------

    #################################################################
    # create fig01                                                  #
    #################################################################

    xlims = (120000, 10000)
    zoom_lim = (40000, 31000)

    colors = {'theta': tc.tol_cset('muted')[5],
              'PIP' : tc.tol_cset('muted')[7],
              'PaTh' : tc.tol_cset('muted')[3],
              'benthic': tc.tol_cset('muted')[4],
              'd18o': tc.tol_cset('muted')[0]}


    width = 165 / 25.4
    height = 180 / 25.4

    fig = plt.figure(figsize=(width, height))
    width_ratios = [0.1, 1, 0.1]
    height_ratios = [1] *6+ [0.2, 1] + [1] * 19
    gs = fig.add_gridspec(27, 3,
                          hspace=0.5,
                          left=0.1,
                          bottom=0.08,
                          top=0.92,
                          right=0.92,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios)

    # add axis
    ax_d18o = fig.add_subplot(gs[0:3, :])
    ax_sim0 = fig.add_subplot(gs[3:6, :])
    ax_theta0 = ax_sim0.twinx()
    ax_frame1 = fig.add_subplot(gs[0:6, :])
    ax_d18o_zoom = fig.add_subplot(gs[8:11, 1])
    ax_PIP = fig.add_subplot(gs[10:13, 1])
    ax_PaTh = fig.add_subplot(gs[12:15, 1])
    #ax_SST = fig.add_subplot(gs[14:17, 1])
    ax_benthic = fig.add_subplot(gs[14:17, 1])
    ax_frame2 = fig.add_subplot(gs[8:17, 1], zorder = -20)
    ax_sim0_zoom = fig.add_subplot(gs[18:21, 1])
    ax_sim3_zoom = fig.add_subplot(gs[20:23, 1])
    ax_sim1_zoom = fig.add_subplot(gs[22:25, 1])
    ax_sim2_zoom = fig.add_subplot(gs[24:27, 1])
    ax_frame3 = fig.add_subplot(gs[18:27, 1], zorder = -20)

    # ----------------------------------------------------------------

    # ax_d18o
    make_patch_spines_invisible(ax_d18o)
    ax_d18o.plot(d18o_age, d18o, color=colors['d18o'], lw=0.8)
    ax_d18o.yaxis.set_ticks_position('left')
    ax_d18o.yaxis.set_label_position('left')
    ax_d18o.spines['left'].set_visible(True)
    ax_d18o.spines['top'].set_visible(True)
    ax_d18o.xaxis.set_visible(True)
    ax_d18o.xaxis.set_ticks_position('top')
    ax_d18o.xaxis.set_label_position('top')
    #ax_d18o.tick_params(axis='x', which='major', direction='inout')
    ax_d18o.set_xticks(np.arange(10000, 130000, 10000))
    ax_d18o.set_xticks(np.arange(10000, 130000, 5000), minor=True)
    ax_d18o.set_yticks([-45, -40, -35])
    ax_d18o.set_yticks([-47.5, -42.5, -37.5, -32.5], minor = True)
    ax_d18o.set_xticklabels((ax_d18o.get_xticks()/1000).astype(np.int32))
    ax_d18o.set_xlim(xlims)
    ax_d18o.set_ylabel('$\delta^{18}$O$_{\mathrm{NGRIP}}$ [\u2030]', color=colors['d18o'])
    ax_d18o.set_xlabel('age [kyr b2k]', labelpad = 12)

    ax_d18o.annotate('(a)', (0.96, 0.8), xycoords='axes fraction')

    GI = True
    for start, end in zip(prox_transitions['age'].values[::-1],
                          np.append(prox_transitions['age'].values[-2::-1], 0)):
        if GI:
            #color = 'C3'
            GI = not GI

        else:
            #color = 'C0'
            ax_d18o.axvspan(start,
                            end,
                            alpha=0.2,
                            edgecolor=None,
                            facecolor='slategray',
                            zorder=1)
            GI = not GI


    # ----------------------------------------------------------------

    # ax_sim0
    make_patch_spines_invisible(ax_sim0)
    ax_sim0.plot(-sim['time'], sim['theta'], color=colors['theta'], lw=0.8)
    ax_sim0.yaxis.set_ticks_position('right')
    ax_sim0.yaxis.set_label_position('right')
    ax_sim0.spines['right'].set_visible(True)
    ax_sim0.spines['right'].set_position(('data', 15000))
    ax_sim0.spines['bottom'].set_visible(True)
    ax_sim0.spines['bottom'].set_bounds(low = 115000, high = 15000)
    ax_sim0.xaxis.set_visible(True)
    ax_sim0.set_xticks(ax_d18o.get_xticks()[1:-1])
    ax_sim0.set_xticks(ax_d18o.get_xticks(minor = True), minor = True)
    ax_sim0.set_xticklabels([])
    ax_sim0.set_xlim(xlims)
    ax_sim0.set_ylim(ax_sim0.get_ylim()[::-1])
    ax_sim0.set_yticks([2,1.5,1])
    ax_sim0.set_yticks([1.75,1.25], minor = True)
    #ax_sim0.set_yticklabels(ax_sim0.get_yticks()*25)
    ax_sim0.set_ylabel(r'$\theta$', color=colors['theta'])
    ax_sim0.annotate('(b)', (0.05, 0.1), xycoords='axes fraction')

    GI = True
    for start, end in zip(sim_transitions['age'].values[::-1],
                          np.append(sim_transitions['age'].values[-2::-1], 0)):

        if GI:
            #color = 'C3'
            GI = not GI
        else:
            #color = 'C0'
            ax_sim0.axvspan(start,
                            end,
                            alpha=0.2,
                            edgecolor=None,
                            facecolor='slategray',
                            zorder=1)
            GI = not GI


    # ----------------------------------------------------------------
    
    make_patch_spines_invisible(ax_theta0)
    ax_theta0.plot(-sim['time'], sim['theta0'], color='k', lw = .8, alpha = .6)
    ax_theta0.set_xlim(xlims)
    ax_theta0.set_yticks(np.arange(1, 2.1, 0.5))
    ax_theta0.set_yticks(np.arange(1, 2.1, 0.1), minor=True)
    ax_theta0.yaxis.set_ticks_position('left')
    ax_theta0.yaxis.set_label_position('left')
    ax_theta0.spines['left'].set_visible(True)
    ax_theta0.spines['left'].set_position(('data', 115000))
    ax_theta0.spines['right'].set_position(('data', 15000))
    ax_theta0.spines['bottom'].set_bounds(low = 115000, high =15000)
    ax_theta0.set_ylim(ax_sim0.get_ylim())
    ax_theta0.set_ylabel(r'$\theta_{0}$',
                         color='k')

    ax_theta0.xaxis.set_visible(False)


    # ----------------------------------------------------------------
    # ax_frame1
    make_patch_spines_invisible(ax_frame1)
    ax_frame1.set_zorder(-1)
    ax_frame1.set_xlim(xlims)
    ax_frame1.xaxis.set_visible(False)
    ax_frame1.yaxis.set_visible(False)
    ax_frame1.axvline(zoom_lim[0],
                      color='slategray',
                      zorder=3,
                      lw=1.)
    ax_frame1.axvline(zoom_lim[1],
                      color='slategray',
                      zorder=1,
                      lw=1.)

    # ----------------------------------------------------------------

    # ax_d18o_zoom
    make_patch_spines_invisible(ax_d18o_zoom)
    ax_d18o_zoom.plot(d18o_age, d18o, color=colors['d18o'])
    ax_d18o_zoom.spines['right'].set_visible(True)
    ax_d18o_zoom.spines['left'].set_visible(True)
    ax_d18o_zoom.spines['top'].set_visible(True)
    ax_d18o_zoom.xaxis.set_ticks_position('top')
    ax_d18o_zoom.xaxis.set_label_position('top')
    ax_d18o_zoom.set_xticks(np.arange(10000, 50000, 1000))
    ax_d18o_zoom.set_xticklabels(
        (ax_d18o_zoom.get_xticks()/1000).astype(np.int32))
    ax_d18o_zoom.set_xlim(zoom_lim)
    ax_d18o_zoom.set_ylabel('$\delta^{18}$O$_{\mathrm{NGRIP}}$ [\u2030]',
                            color=colors['d18o'],
                            verticalalignment='center')
    ax_d18o_zoom.yaxis.set_label_coords(-0.2, 0.5)
    ax_d18o_zoom.annotate('(c)', (0.02, 0.7), xycoords='axes fraction')

    # ax_PIP
    make_patch_spines_invisible(ax_PIP)
    ax_PIP.scatter(PIP_age, PIP,
                   color=colors['PIP'],
                   marker='v',
                   s=8,
                   lw=1)
    ax_PIP.set_xlim(zoom_lim)
    ax_PIP.xaxis.set_visible(False)
    ax_PIP.spines['right'].set_visible(True)
    ax_PIP.spines['left'].set_visible(True)
    ax_PIP.yaxis.set_ticks_position('right')
    ax_PIP.yaxis.set_label_position('right')
    ax_PIP.set_ylabel('MD95-2010 \n PIP$_{25}$', color=colors['PIP'],
                      verticalalignment='center')
    ax_PIP.yaxis.set_label_coords(1.17, 0.6)
    ax_PIP.set_ylim(0.9, 0.1)
    ax_PIP.annotate('(d)', (0.95, 0.65), xycoords='axes fraction')

    # ax_PaTh
    make_patch_spines_invisible(ax_PaTh)
    ax_PaTh.scatter(PaTh_age, PaTh,
                    color=colors['PaTh'],
                    marker='x',
                    lw=1.1,
                    s=11)
    ax_PaTh.set_xlim(zoom_lim)
    ax_PaTh.xaxis.set_visible(False)
    ax_PaTh.spines['right'].set_visible(True)
    ax_PaTh.spines['left'].set_visible(True)
    # ax_PaTh.yaxis.set_ticks_position('right')
    # ax_PaTh.yaxis.set_label_position('right')
    ax_PaTh.set_ylabel('CDH19 \n Pa$^{231}$/$^{230}$Th', color=colors['PaTh'],
                       verticalalignment='center')
    ax_PaTh.yaxis.set_label_coords(-0.18, 0.4)
    ax_PaTh.set_ylim(ax_PaTh.get_ylim()[::-1])
    ax_PaTh.set_yticks([0.08, 0.06, 0.04])
    ax_PaTh.annotate('(e)', (0.02, 0.7), xycoords='axes fraction')


    # ax_benthic
    make_patch_spines_invisible(ax_benthic)
    ax_benthic.scatter(benthic_age, benthic,
                       color=colors['benthic'],
                       marker='1', s=10, lw=1)
    ax_benthic.set_xlim(zoom_lim)
    ax_benthic.spines['right'].set_visible(True)
    ax_benthic.spines['left'].set_visible(True)
    # ax_benthic.spines['bottom'].set_visible(True)
    ax_benthic.xaxis.set_visible(False)
    ax_benthic.set_ylabel('MD99-2284 \n C. neoteretis \n $\delta^{18}$O [\u2030 VPDB]',
                          color=colors['benthic'],
                          verticalalignment='center')
    ax_benthic.yaxis.set_label_coords(1.1, 0.5)
    ax_benthic.yaxis.set_ticks_position('right')
    ax_benthic.yaxis.set_label_position('right')
    # ax_benthic.set_ylim(ax_benthic.get_ylim()[::-1])
    # ax_benthic.set_ylim(ax_benthic.get_ylim()[::-1])
    ax_benthic.annotate('(f)', (0.95, 0.7), xycoords='axes fraction')
    ax_benthic.set_yticks([3.5, 4.5, 5.5])
    # ax_frame2

    make_patch_spines_invisible(ax_frame2)
    ax_frame2.xaxis.set_visible(False)
    ax_frame2.yaxis.set_visible(False)
    ax_frame2.set_xlim(zoom_lim)

    GI = True
    for start, end in zip(prox_transitions['age'].values[::-1],
                          np.append(prox_transitions['age'].values[-2::-1], 0)):
        if not GI:
            ax_frame2.axvspan(start,
                              end,
                              alpha=0.2,
                              edgecolor=None,
                              facecolor='slategray',
                              zorder=1)

        GI = not GI

    # ----------------------------------------------------------------

    # PLOTTING SIMULATED DATA

    # ax_sim0_zoom
    make_patch_spines_invisible(ax_sim0_zoom)
    ax_sim0_zoom.plot(-sim['time'], sim['theta'], color=colors['theta'])
    ax_sim0_zoom.set_ylim(ax_sim0_zoom.get_ylim()[::-1])
    ax_sim0_zoom.spines['right'].set_visible(True)
    ax_sim0_zoom.spines['left'].set_visible(True)
    ax_sim0_zoom.set_yticks([2,1.5,1])
    ax_sim0_zoom.set_yticks([1.75,1.25], minor = True)
    ax_sim0_zoom.set_xlim(zoom_lim)
    ax_sim0_zoom.xaxis.set_visible(False)
    ax_sim0_zoom.set_ylabel(r'$\theta$', color=colors['theta'],
                            verticalalignment='center')
    ax_sim0_zoom.yaxis.set_label_coords(-0.2, 0.5)
    ax_sim0_zoom.annotate('(g)', (0.02, 0.7), xycoords='axes fraction')

    # ax_sim3_zoom
    make_patch_spines_invisible(ax_sim3_zoom)
    ax_sim3_zoom.plot(-sim['time'], sim['I'], color=colors['PIP'])
    ax_sim3_zoom.set_xlim(zoom_lim)
    ax_sim3_zoom.xaxis.set_visible(False)
    ax_sim3_zoom.spines['right'].set_visible(True)
    ax_sim3_zoom.spines['left'].set_visible(True)
    ax_sim3_zoom.set_ylabel('$I$', color=colors['PIP'],
                            verticalalignment='center')
    ax_sim3_zoom.yaxis.set_label_coords(1.15, 0.5)
    # ax_sim3_zoom.set_ylim(ax_sim3_zoom.get_ylim()[::-1])
    ax_sim3_zoom.set_ylim(6, -10)
    ax_sim3_zoom.set_yticks([5,0,-5])
    # ax_sim3_zoom.set_yticks([2.5,-2.5, -7.5], minor = True)
    ax_sim3_zoom.yaxis.set_ticks_position('right')
    ax_sim3_zoom.yaxis.set_label_position('right')
    ax_sim3_zoom.annotate('(h)', (0.02, 0.7), xycoords='axes fraction')

    # ax_sim3_zoom.set_ylim(ax_sim2_zoom.get_ylim()[::-1])

    # ax_sim1_zoom
    make_patch_spines_invisible(ax_sim1_zoom)
    ax_sim1_zoom.plot(-sim['time'], sim['T'] - sim['S'], color=colors['PaTh'])
    ax_sim1_zoom.set_xlim(zoom_lim)
    ax_sim1_zoom.xaxis.set_visible(False)
    ax_sim1_zoom.spines['right'].set_visible(True)
    ax_sim1_zoom.spines['left'].set_visible(True)
    # ax_sim1_zoom.yaxis.set_ticks_position('right')
    # ax_sim1_zoom.yaxis.set_label_position('right')
    ax_sim1_zoom.set_ylabel('$q$', color=colors['PaTh'],
                            verticalalignment='center')
    ax_sim1_zoom.yaxis.set_label_coords(-0.2, 0.5)
    ax_sim1_zoom.annotate('(i)', (0.02, 0.7), xycoords='axes fraction')
    # ax_sim1_zoom.set_ylim(ax_sim1_zoom.get_ylim()[::-1])

    # ax_sim2_zoomp
    make_patch_spines_invisible(ax_sim2_zoom)
    ax_sim2_zoom.plot(-sim['time'], sim['T'], color=colors['benthic'])
    # ax_sim2_zoom.xaxis.set_visible(False)
    ax_sim2_zoom.spines['right'].set_visible(True)
    ax_sim2_zoom.spines['left'].set_visible(True)
    ax_sim2_zoom.spines['bottom'].set_visible(True)
    ax_sim2_zoom.set_ylabel('$T$', color=colors['benthic'],
                            verticalalignment='center')
    ax_sim2_zoom.yaxis.set_label_coords(1.17, 0.5)
    ax_sim2_zoom.yaxis.set_ticks_position('right')
    # ax_sim2_zoom.yaxis.set_label_position('right')
    ax_sim2_zoom.annotate('(j)', (0.02, 0.7), xycoords='axes fraction')
    ax_sim2_zoom.set_xlabel('age [kyr b2k]', labelpad=12)
    ax_sim2_zoom.set_xticks(np.arange(10000, 50000, 1000))
    ax_sim2_zoom.set_xticklabels(
        (ax_sim2_zoom.get_xticks()/1000).astype(np.int32))
    ax_sim2_zoom.set_xlim(zoom_lim)
    ax_sim2_zoom.set_ylim((0.3,0.85))
    ax_sim2_zoom.set_yticks([0.4,0.6,0.8])
    # ax_sim2_zoom.set_yticklabels(ax_sim2_zoom.get_yticks() * 25)
    # ax_sim2_zoom.set_ylim(ax_sim2_zoom.get_ylim()[::-1])

    # ax_benthic
    # make_patch_spines_invisible(ax_benthic)
    # ax_benthic.scatter(benthic_age, benthic, color ='C6', marker = '1')
    # ax_benthic.set_xlim(zoom_lim)
    # ax_benthic.spines['right'].set_visible(True)
    # ax_benthic.spines['left'].set_visible(True)
    # ax_benthic.spines['bottom'].set_visible(True)
    # ax_benthic.set_ylabel('benthic', color ='C6')
    # ax_benthic.yaxis.set_ticks_position('left')
    # ax_benthic.yaxis.set_label_position('left')
    # ax_benthic.set_ylim(ax_benthic.get_ylim()[::-1])

    # ax_frame2

    make_patch_spines_invisible(ax_frame3)

    GI = True
    for start, end in zip(sim_transitions['age'].values[::-1],
                          np.append(sim_transitions['age'].values[-2::-1], 0)):
        if not GI:
            ax_frame3.axvspan(start,
                              end,
                              alpha=0.2,
                              edgecolor=None,
                              facecolor='slategray',
                              zorder=1)

        GI = not GI

    ax_frame3.xaxis.set_visible(False)
    ax_frame3.yaxis.set_visible(False)
    ax_frame3.set_xlim(zoom_lim)

    # draw zoom lines
    xy0 = (zoom_lim[0], ax_sim0.get_ylim()[0])
    xy1 = (zoom_lim[0], ax_d18o_zoom.get_ylim()[1])
    con1 = ConnectionPatch(xy0, xy1, 'data', 'data',
                           ax_sim0, ax_d18o_zoom,
                           ls='solid',
                           zorder=1,
                           color='slategray',
                           lw=1.)
    ax_d18o_zoom.add_artist(con1)

    xy0 = (zoom_lim[1], ax_sim0.get_ylim()[0])
    xy1 = (zoom_lim[1], ax_d18o_zoom.get_ylim()[1])
    con1 = ConnectionPatch(xy0, xy1, 'data', 'data',
                           ax_sim0, ax_d18o_zoom,
                           ls='solid',
                           zorder=1,
                           color='slategray',
                           lw=1.)
    ax_d18o_zoom.add_artist(con1)

    directory = dirname(params, sub='figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)

    fig.savefig(directory + '/fig01_n%i.pdf' % (ntraj), format='pdf')
    fig.savefig(directory + '/fig01_n%i.png' % (ntraj), dpi=300)
    # plt.close()

    #################################################################

    # fig, ax = plt.subplots()
    # ax.plot(time, simulation[:,1])
    # ax.plot(time, simulation[:,2], color = 'C1')
    # ax.plot(time, simulation[:,1]-simulation[:,2], color = colors['PIP'])
