import numpy as np
import fnmatch
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from plot_functions import binning
import pandas as pd
from scipy.interpolate import interp1d
from os import mkdir
from os.path import isdir, exists
from glob import glob
from miscellanious import dirname
from miscellanious import check_results
from plot_functions import make_patch_spines_invisible
from plot_functions import NGRIP_stadial_mask
from transient_trajectories import compute_trans_traj
from transition_detection import transitions_proxy_record
from transition_detection import detect_transitions_sim
from create_dur_stats import trans_stats
from create_dur_stats import compute_rw_stats
import tol_colors as tc

colors ={'theta': tc.tol_cset('muted')[5],
         'PIP' : tc.tol_cset('muted')[7],
         'PaTh' : tc.tol_cset('muted')[3],
         'benthic': tc.tol_cset('muted')[4],
         'd18o': tc.tol_cset('muted')[0]}


def plot_trans_traj(params, ntraj=1, recompute=False):

    #############################################################
    # import simulation data                                    #
    #############################################################

    directory = dirname(params)
    filename = 'trans_traj_5y_n%i.csv' % (ntraj)
    sim_name, sim_exists = check_results(params, filename)

    if not sim_exists or recompute:
        print(r'computing trans traj')
        compute_trans_traj(params, ntraj=ntraj)

    sim = pd.read_csv(sim_name)

    #############################################################
    # import proxy data                                         #
    #############################################################

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
    prox_trans, prox_dur = transitions_proxy_record()

    # ----------------------------------------------------------------

    #################################################################
    # create statistics from the proxy time series and the selected #
    # simulated time series                                         # 
    #################################################################

    w_pos, GI_prox_mean, GS_prox_mean, GI_prox_freq, GS_prox_freq = compute_rw_stats(
         prox_trans['age'], prox_trans['start of'], prox_trans['duration'])

    sim_trans, sim_dur = detect_transitions_sim(sim, params)
    w_pos, GI_sim_mean, GS_sim_mean, GI_sim_freq, GS_sim_freq = compute_rw_stats(
        sim_trans['age'], sim_trans['start of'], sim_trans['duration'])

    if GI_sim_mean is np.nan:
        print('trajectory %i exhibits a climate period > 20kyr and is omitted' % (ntraj))
        return

    # statsfile = glob(dirname(params) + '/stats_n*.csv')
    # data_files = glob(directory + '/trans_traj_5y*.csv')

    # if len(statsfile) == 0 or recompute:

    #     sims_excluded = 0
    #     GI_sim_means = np.zeros((len(data_files), len(w_pos)))
    #     GS_sim_means = np.zeros((len(data_files), len(w_pos)))
    #     GS_sim_freqs = np.zeros((len(data_files), len(w_pos)))
    #     GI_sim_freqs = np.zeros((len(data_files), len(w_pos)))

    #     for n, f in enumerate(data_files):
    #         print(n)

    #         temp = pd.read_csv(f)
    #         temp_trans, temp_dur = detect_transitions_sim(temp, params)
    #         w_pos, GI_temp_mean, GS_temp_mean, GI_temp_freq, GS_temp_freq = compute_rw_stats(
    #             temp_trans['age'], temp_trans['start of'], temp_trans['duration'])

    #         if GI_temp_mean is np.nan:
    #             sims_excluded += 1

    #         GI_sim_means[n] = GI_temp_mean
    #         GS_sim_means[n] = GS_temp_mean
    #         GS_sim_freqs[n] = GS_temp_freq
    #         GI_sim_freqs[n] = GI_temp_freq

    #     GI_sim_mean_stats = np.nanpercentile(GI_sim_means, [5, 50, 95], axis=0)
    #     GS_sim_mean_stats = np.nanpercentile(GS_sim_means, [5, 50, 95], axis=0)
    #     GI_sim_freq_stats = np.nanpercentile(GI_sim_freqs, [5, 50, 95], axis=0)
    #     GS_sim_freq_stats = np.nanpercentile(GS_sim_freqs, [5, 50, 95], axis=0)

    #     columns = ['age',
    #                'GI_sim_dur_05', 'GI_sim_dur_50', 'GI_sim_dur_95',
    #                'GS_sim_dur_05', 'GS_sim_dur_50', 'GS_sim_dur_95',
    #                'GI_sim_freq_05', 'GI_sim_freq_50', 'GI_sim_freq_95',
    #                'GS_sim_freq_05', 'GS_sim_freq_50', 'GS_sim_freq_95']
    #     stats = pd.DataFrame(np.concatenate((w_pos[None, :],
    #                                          GI_sim_mean_stats,
    #                                          GS_sim_mean_stats,
    #                                          GI_sim_freq_stats,
    #                                          GS_sim_freq_stats), axis=0).T,
    #                          columns=columns)
    #     stats.to_csv(dirname(params) + '/stats_n%i.csv' %
    #                  (100-sims_excluded), index=False)

    # else:
    stats = trans_stats(params, recompute = True)
    # stats = pd.read_csv(directory + '/duration_stats_995.csv')
    # ----------------------------------------------------------------

    #################################################################
    # create figure                                                 #
    #################################################################

    width = 114 / 25.4
    height = 80 / 25.4

    # width = 114 / 25.4
    # height = 114 / 25.4
    fig = plt.figure(figsize=(width, height))

    height_ratios = [0] * 3 + [0] + [4, 1, 4, 1, 4, 0, 0]
    gs = fig.add_gridspec(11, 1,
                          hspace=0,
                          left=0.06,
                          bottom=0.15,
                          top=0.95,
                          right=0.95,
                          height_ratios=height_ratios)

    # ax_d18o = fig.add_subplot(gs[2])
    # ax_LR04 = fig.add_subplot(gs[0])
    # ax_theta0 = ax_LR04.twinx()
    # ax_sim0 = fig.add_subplot(gs[1])
    ax_GI_dur = fig.add_subplot(gs[4])
    ax_GS_dur = fig.add_subplot(gs[6])
    ax_GI_freq = fig.add_subplot(gs[8])
    # ax_GS_freq = fig.add_subplot(gs[10])

    xlims = (115000, 15000)
    xticks = np.arange(10000, 121000, 10000)
    lw = 0.5

    # ax_d18o
    # make_patch_spines_invisible(ax_d18o)
    # ax_d18o.plot(d18o_age, d18o, color=colors['d18o'], lw=0.8)
    # ax_d18o.yaxis.set_ticks_position('left')
    # ax_d18o.yaxis.set_label_position('left')
    # ax_d18o.spines['bottom'].set_visible(False)
    # ax_d18o.spines['top'].set_visible(False)
    # ax_d18o.spines['top'].set_visible(True)
    # ax_d18o.set_xticklabels([])
    #ax_d18o.set_xticks(xticks)
    # ax_d18o.set_xticklabels((xticks/1000).astype(int))
    # ax_d18o.set_xticklabels([])
    # ax_d18o.set_xlim(xlims)
    # ax_d18o.set_ylabel('$\delta^{18}$O$_{\mathrm{NGRIP}}$ [\u2030]',
    #                    color=colors['d18o'], y=1.4, ha='right')
    # ylabel_coords = ax_d18o.get_label_coords()
    # ax_d18o.set_label_coords(ylabel_coords[0],0.6)
    #ax_d18o.annotate('(c)', (0.94, 0.7), xycoords='axes fraction')
    # GI = True
    # for start, end in zip(prox_trans['age'].values[::-1],
    #                       np.append(prox_trans['age'].values[-2::-1], 0)):
    #     if GI:
    #         #color = colors['d18o']
    #         continue
    #     else:
    #         #color = 'C0'
    #         ax_d18o.axvspan(start,
    #                         end,
    #                         alpha=0.2,
    #                         edgecolor=None,
    #                         facecolor='slategray',
    #                         zorder=1)
    #     GI = not GI

# ----------------------------------------------------------------

    # ax_LR04
    # make_patch_spines_invisible(ax_LR04)
    # ax_LR04.plot(-sim['time'], sim['LR04'], color='k')
    # ax_LR04.xaxis.set_visible(True)
    # ax_LR04.xaxis.set_ticks_position('top')
    # ax_LR04.xaxis.set_label_position('top')
    # ax_LR04.set_xlabel('age [kyr b2k]', labelpad=10)
    # ax_LR04.set_xticks(xticks)
    # ax_LR04.set_xticklabels((xticks/1000).astype(int))
    # # ax_LR04.set_xticklabels([])
    # ax_LR04.tick_params(axis='x', which='major', direction='out')
    # # ax_LR04.spines['bottom'].set_visible(False)
    # # ax_LR04.spines['right'].set_visible(True)
    # ax_LR04.set_xlim(xlims)
    # ax_LR04.set_yticks([-2, 0, 2])
    # ax_LR04.set_yticks(np.arange(-2, 2, 0.5), minor=True)

    # ax_LR04.xaxis.set_visible(True)
    # ax_LR04.set_ylim(ax_LR04.get_ylim()[::-1])
    # ax_LR04.set_ylabel(r'$\delta^{18}$O$^{*}_{\mathrm{LR04}}$ [n.u.]',
    #                    color='k', y=-0.4, ha='left')

    # ax_LR04.annotate('(a)', (0.94, 0.7), xycoords='axes fraction')

    # # ax_LR04
    # make_patch_spines_invisible(ax_theta0)
    # ax_theta0.plot(-sim['time'], sim['theta0'], color='None')
    # ax_theta0.set_xlim(xlims)
    # ax_theta0.set_yticks(np.arange(1, 2.1, 0.5))
    # ax_theta0.set_yticks(np.arange(1, 2.1, 0.1), minor=True)
    # LR04_ylim = ax_LR04.get_ylim()
    # a, b = np.polyfit(sim['LR04'], sim['theta0'], 1)
    # ax_theta0.set_ylim((LR04_ylim[0] * a + b, LR04_ylim[1] * a + b))
    # ax_theta0.set_ylabel(r'$\theta_{0}$',
    #                      color='k')

    #ax_theta0.set_yticks(np.arange(1.25, 1.65, 0.1))
    # ax_theta0.xaxis.set_visible(False)

    # #ax_theta0.spines['right'].set_position(('axes', 0.97))
    # #ax_theta0.tick_params(axis='y', direction='in', pad=-25)

    # # ax_sim0
    # # make_patch_spines_invisible(ax_sim0)
    # ax_sim0.plot(-sim['time'], sim['theta'], color=colors['theta'], lw=0.8)
    # ax_sim0.yaxis.set_ticks_position('right')
    # ax_sim0.yaxis.set_label_position('right')
    # # ax_sim0.spines['top'].set_visible(False)
    # # ax_sim0.spines['bottom'].set_visible(False)
    # # ax_sim0.spines['right'].set_position(('axes', .985))
    # # ax_d18o.spines['left'].set_visible(True)
    # ax_sim0.xaxis.set_visible(False)
    # ax_sim0.set_xticks(xticks)
    # ax_sim0.set_xticklabels((xticks/1000).astype(int))
    # ax_sim0.set_yticks(np.arange(2.0, 0.6, -0.5))
    # #ax_sim0.set_yticks(np.arange(0.5, 2.1, 0.1), minor = True)
    # ax_sim0.set_xlim(xlims)
    # ax_sim0.set_ylim(ax_sim0.get_ylim()[::-1])
    # ax_sim0.set_ylabel(r'$\theta$', color=colors['theta'])
    # ax_sim0.annotate('(b)', (0.014, 0.15), xycoords='axes fraction')

    # GI = True
    # for start, end in zip(sim_trans['age'].values[::-1],
    #                       np.append(sim_trans['age'].values[-2::-1], 0)):

    #     if GI:
    #         color = colors['d18o']
    #     else:
    #         color = 'C0'
    #     ax_sim0.axvspan(start,
    #                     end,
    #                     alpha=0.2,
    #                     edgecolor=None,
    #                     facecolor=color,
    #                     zorder=1)
    #     GI = not GI

    # ax_GI_dur
    make_patch_spines_invisible(ax_GI_dur)
    ax_GI_dur.plot(w_pos, GI_prox_mean, color=colors['d18o'], label = r'$\delta^{18}O_{\mathrm{NGRIP}}$')
    ax_GI_dur.plot(w_pos, GI_sim_mean, color=colors['theta'], label = r'$\theta$')

    ax_GI_dur.plot(stats['age'].values, stats['GI_sim_dur_50'].values,
                   color='k',
                   zorder=-1,
                   label = 'median')

    ax_GI_dur.fill_between(stats['age'].values,
                           stats['GI_sim_dur_25'].values,
                           stats['GI_sim_dur_75'].values,
                           color='slategray', lw=lw,
                           alpha=0.5,
                           zorder=-2, label = 'IQR')
    ax_GI_dur.fill_between(stats['age'].values,
                           stats['GI_sim_dur_05'].values,
                           stats['GI_sim_dur_95'].values,
                           color='slategray', lw=lw,
                           alpha=0.2,
                           zorder=-5,
                           label = 'CI-90')

    ax_GI_dur.plot(stats['age'].values, stats['GI_sim_dur_25'].values,
                   color='slategray',
                   zorder=-1,
                   lw = 1)
    ax_GI_dur.plot(stats['age'].values, stats['GI_sim_dur_75'].values,
                   color='slategray',
                   zorder=-1,
                   lw = 1)


    #ax_GI_dur.set_ylim(0, 6000)
    ax_GI_dur.set_xlim(xlims)
    # ax_dur.set_xticklabels(xticklabels)
    ax_GI_dur.set_ylabel(
        r'$\langle\tau_{\mathrm{inter}}\rangle_{20\;\mathrm{kyr}}\;[y]$')
    # ax_GI_dur.xaxis.set_visible(False)
    ax_GI_dur.set_xticks(np.arange(30000, 110000, 10000))
    ax_GI_dur.set_xticklabels([])

    ax_GI_dur.set_xlim(xlims)
    ax_GI_dur.set_yticks(np.arange(0, 7000, 2000))
    ax_GI_dur.set_yticks(np.arange(1000, 7000, 2000), minor=True)
    ax_GI_dur.spines['left'].set_visible(True)
    ax_GI_dur.spines['right'].set_visible(True)
    ax_GI_dur.spines['bottom'].set_visible(True)
    ax_GI_dur.spines['bottom'].set_bounds(low=105200,
                                          high=24800)
    ax_GI_dur.yaxis.set_ticks_position('left')
    ax_GI_dur.yaxis.set_label_position('left')
    ax_GI_dur.annotate('(a)', (0.85, 0.8), xycoords='axes fraction')
    ax_GI_dur.spines['right'].set_position(('data', 24800))
    ax_GI_dur.spines['left'].set_position(('data', 105200))
    ax_GI_dur.tick_params(axis='y',
                          which='both',
                          direction='out',
                          left=True,
                          right=True)
    legend = ax_GI_dur.legend(bbox_to_anchor = (0.4, 1),
                              loc = 'upper left',
                              frameon = False,
                              ncol = 2,
                              fontsize = 8)

    legend.legendHandles[3].set_x(- 10)
    #legend.legendHandles[3].set_y(- 10)
    legend.texts[3].set_x(-10)
    #legend.texts[3].set_y(-10)
    legend.legendHandles[4].set_x(- 10)
    #legend.legendHandles[4].set_y(- 10)
    legend.texts[4].set_x(-10)
    #legend.texts[4].set_y(-10)
    xdata = legend.legendHandles[2].get_xdata()
    legend.legendHandles[2].set_xdata(xdata + 70)
    legend.texts[2].set_x(330)

    # ax_GS_dur
    make_patch_spines_invisible(ax_GS_dur)
    ax_GS_dur.plot(w_pos, GS_prox_mean, color=colors['d18o'])
    ax_GS_dur.plot(w_pos, GS_sim_mean, color=colors['theta'])
    ax_GS_dur.fill_between(stats['age'].values,
                           stats['GS_sim_dur_25'].values,
                           stats['GS_sim_dur_75'].values,
                           color = 'slategray', lw = lw,
                           alpha = 0.5,
                           zorder = -2)
    ax_GS_dur.fill_between(stats['age'].values,
                           stats['GS_sim_dur_05'].values,
                           stats['GS_sim_dur_95'].values,
                           color = 'slategray', lw = lw,
                           alpha = 0.2,
                           zorder = -5)

    ax_GS_dur.plot(stats['age'].values, stats['GS_sim_dur_50'].values, 
                   color = 'k',
                   zorder = -1)
    ax_GS_dur.plot(stats['age'].values, stats['GS_sim_dur_75'].values, 
                   color = 'slategray',
                   zorder = -1,
                   lw = 1)
    ax_GS_dur.plot(stats['age'].values, stats['GS_sim_dur_25'].values, 
                   color = 'slategray',
                   zorder = -1,
                   lw = 1)

    ax_GS_dur.set_ylabel(r'$\langle\tau_{\mathrm{stadial}}\rangle_{20\;\mathrm{kyr}}\;[y]$')

    # ax_GS_dur.spines['top'].set_visible(False)
    #ax_GS_dur.spines['bottom'].set_visible(False)
    #ax_GS_dur.set_ylim(0, 4000)
    ax_GS_dur.spines['left'].set_visible(True)
    ax_GS_dur.spines['right'].set_visible(True)
    ax_GS_dur.spines['bottom'].set_visible(True)
    ax_GS_dur.spines['bottom'].set_bounds(low = 105100,
                                          high = 24800)
    ax_GS_dur.set_xlim(xlims)
    ax_GS_dur.set_yticks(np.arange(0,7000,2000))
    ax_GS_dur.set_yticks(np.arange(1000,8000,2000), minor = True)
    ax_GS_dur.yaxis.set_ticks_position('right')
    ax_GS_dur.yaxis.set_label_position('right')
    #ax_GS_dur.xaxis.set_visible(False)
    ax_GS_dur.set_xticks(np.arange(30000, 110000, 10000))
    ax_GS_dur.set_xticklabels([])

    ax_GS_dur.annotate('(b)', (0.12, 0.8), xycoords='axes fraction')
    ax_GS_dur.spines['right'].set_position(('data', 24800))
    ax_GS_dur.spines['left'].set_position(('data', 105200))
    ax_GS_dur.tick_params(axis='y',
                          which = 'both',
                          direction='out',
                          left = True,
                          right = True)


    make_patch_spines_invisible(ax_GI_freq)
    ax_GI_freq.plot(w_pos, GI_prox_freq, color=colors['d18o'])
    ax_GI_freq.plot(w_pos, GI_sim_freq, color=colors['theta'])
    ax_GI_freq.fill_between(stats['age'].values,
                            stats['GI_sim_freq_25'].values,
                            stats['GI_sim_freq_75'].values,
                            color = 'slategray', lw = lw,
                            alpha = 0.5,
                            zorder = -2)
    ax_GI_freq.fill_between(stats['age'].values,
                            stats['GI_sim_freq_05'].values,
                            stats['GI_sim_freq_95'].values,
                            color = 'slategray', lw = lw,
                            alpha = 0.2,
                            zorder = -5)
    
    ax_GI_freq.plot(stats['age'].values, stats['GI_sim_freq_50'].values, 
                    color = 'k',
                    zorder = -1)
    ax_GI_freq.plot(stats['age'].values, stats['GI_sim_freq_25'].values, 
                    color = 'slategray',
                    zorder = -1,
                    lw = 1)
    ax_GI_freq.plot(stats['age'].values, stats['GI_sim_freq_75'].values, 
                    color = 'slategray',
                    zorder = -1,
                    lw = 1)

    #ax_GI_freq.spines['bottom'].set_visible(False)
    #ax_GI_freq.spines['top'].set_visible(False)
    #ax_GI_freq.spines['bottom'].set_visible(False)
    ax_GI_freq.set_xlim(xlims)
    ax_GI_freq.set_ylabel(r'$N^{\mathrm{DO}}_{20\;\mathrm{kyr}}$')
    #ax_GI_freq.xaxis.set_visible(False)
    ax_GI_freq.set_xticks(np.arange(30000, 110000, 10000))
    ax_GI_freq.set_xticklabels([])
    ax_GI_freq.yaxis.set_ticks_position('left')
    ax_GI_freq.yaxis.set_label_position('left')
    ax_GI_freq.annotate('(c)', (0.12, 0.8), xycoords='axes fraction')
    ax_GI_freq.spines['left'].set_visible(True)
    ax_GI_freq.spines['right'].set_visible(True)
    ax_GI_freq.spines['bottom'].set_visible(True)
    ax_GI_freq.spines['bottom'].set_bounds(low = 105100,
                                          high = 24800)
    ax_GI_freq.set_yticks(np.arange(0,16,5))
    ax_GI_freq.set_yticks(np.arange(0,16,1), minor = True)
    ax_GI_freq.spines['right'].set_position(('data', 24800))
    ax_GI_freq.spines['left'].set_position(('data', 105200))
    ax_GI_freq.tick_params(axis='y',
                          which = 'both',
                          direction='out',
                          left = True,
                          right = True)
    ax_GI_freq.set_xticks(np.arange(30000, 110000, 10000))
    ax_GI_freq.set_xticklabels((np.arange(30, 110, 10)).astype(int))
    ax_GI_freq.set_xlabel('age [kyr b2k]', labelpad=5)

    # make_patch_spines_invisible(ax_GS_freq)
    # ax_GS_freq.plot(w_pos, GS_prox_freq, color=colors['d18o'])
    # ax_GS_freq.plot(w_pos, GS_sim_freq, color=colors['theta'])
    # ax_GS_freq.fill_between(stats['age'].values,
    #                         stats['GS_sim_freq_05'].values,
    #                         stats['GS_sim_freq_95'].values,
    #                         color = 'slategray', lw = lw,
    #                         alpha = 0.5,
    #                         zorder = -2)
    # ax_GS_freq.plot(stats['age'].values, stats['GS_sim_freq_50'].values,
    #                 color = 'k',
    #                 zorder = -3)
    # ax_GS_freq.plot(stats['age'].values, stats['GS_sim_freq_25'].values,
    #                 color = 'slategray',
    #                 zorder = -3)
    # ax_GS_freq.plot(stats['age'].values, stats['GS_sim_freq_75'].values,
    #                 color = 'slategray',
    #                 zorder = -3)
    # #ax_GS_freq.spines['top'].set_visible(False)
    # ax_GS_freq.set_ylabel(r'$N^{\mathrm{cool}}_{20\;\mathrm{kyr}}$')
    # # ax_GS_freq.yaxis.set_ticks_position('right')
    # # ax_GS_freq.yaxis.set_label_position('right')
    # ax_GS_freq.set_xlabel('age [kyr b2k]', labelpad=5)
    # ax_GS_freq.set_xlim(xlims)
    # ax_GS_freq.tick_params(axis='x', which='major', direction='inout')
    # ax_GS_freq.annotate('(g)', (0.12, 0.8), xycoords='axes fraction')
    # ax_GS_freq.spines['left'].set_visible(True)
    # ax_GS_freq.spines['right'].set_visible(True)
    # ax_GS_freq.spines['bottom'].set_visible(True)
    # ax_GS_freq.spines['bottom'].set_bounds(low = 105100,
    #                                       high = 24800)
    # ax_GS_freq.set_yticks(np.arange(0,16,5))
    # ax_GS_freq.set_yticks(np.arange(0,16,1), minor = True)
    # ax_GS_freq.spines['right'].set_position(('data', 24800))
    # ax_GS_freq.spines['left'].set_position(('data', 105200))
    # ax_GS_freq.tick_params(axis='y',
    #                       which = 'both',
    #                       direction='out',
    #                       left = True,
    #                       right = True)


    
    directory = dirname(params, sub = 'figures')
    if not isdir(directory):
        mkdir(directory)
        print('created' + directory)

    fig.savefig(directory + '/fig06_n%i.png' %(ntraj), dpi=300)
    fig.savefig(directory + '/fig06_n%i.pdf' %(ntraj), format = 'pdf')

    plt.close()
    
    # idx = np.where(np.diff(theta0 <1.375))[0]

    # for i in range(0, len(idx), 2):
    #    ax_sim0.axvspan(time[idx[i]],
    #                    time[idx[i+1]],
    #                    edgecolor=None,
    #                    facecolor='olive',
    #                    alpha = 0.2,
    #                    zorder=-1)
