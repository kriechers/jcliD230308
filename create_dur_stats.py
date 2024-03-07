import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir, exists
from glob import glob
from miscellanious import dirname
from miscellanious import check_results
from transition_detection import transitions_proxy_record
from transition_detection import detect_transitions_sim

def compute_rw_stats(transition_ages, start_of, durations):

    #################################################################
    # compute the average duration GI and GS duration in a 20kyr rw #
    #################################################################

    window_centers = np.arange(105000, 24000, -1000)
    # this results in 81 windows
    
    GI_duration_means = np.zeros_like(window_centers)
    GI_freq = np.zeros_like(window_centers)
    GS_duration_means = np.zeros_like(window_centers)
    GS_freq = np.zeros_like(window_centers)

    for i, c in enumerate(window_centers):
        mask = (c + 10000 > transition_ages) & (c - 10000 < transition_ages)
        if all(~mask):
            print('simulation includes period of 20000 without transitions and is neglected')
            return window_centers, np.nan, np.nan, np.nan, np.nan
        # the mask finds all the transitions, that happen inside
        # the 20 kyr window. We add to this the latest transition
        # before the window, making sure that the climate period
        # at the beginning of the window is still taken into
        # account.
        idx = np.argwhere(transition_ages.values >= c+ 10000)[0]
        # transition ages are increasing with increasing index. 
        mask[idx] = True
        GI_mask = np.array(['GI' in x for x in start_of[mask]])
        GI_duration_means[i] = np.nanmean(durations[mask][GI_mask])
        GI_freq[i] = np.sum(GI_mask)
        GS_mask = np.array(['GS' in x for x in start_of[mask]])
        GS_duration_means[i] = np.mean(durations[mask][GS_mask])
        GS_freq[i] = np.sum(GS_mask)

    return window_centers, GI_duration_means, GS_duration_means, GI_freq, GS_freq




def trans_stats(params, recompute = False):

    stat_list = glob(dirname(params) + '/duration_stats_*.csv')
    
    if (not recompute) and len(stat_list) >0:
        n_list = [x.split('_')[-1][:-4] for x in stat_list]
        max_idx = np.argmax(n_list)
        stats = pd.read_csv(stat_list[max_idx])
        return stats
        
    directory = dirname(params)
    data_files = glob(directory + '/trans_traj_5y*.csv')
    results = np.zeros((len(data_files), 81, 4))
    # 81 is the number of running windows
    # for each running window 4 quantities are stored in results.
    # the mean duration of stadials and interstadials within the
    # running window and the number of stadials and interstadials
    # within the window. 

    for f in data_files: 
        sim = pd.read_csv(f)
        ntraj = f.split('_')[-1].split('.')[0]
        n = int(ntraj[1:])

        #################################################################
        # create statistics from all existing transient trajectories    #
        #################################################################

        sim_trans, sim_dur = detect_transitions_sim(sim, params)

        ### control figure ###

        fig, ax = plt.subplots()
        ax.plot(sim['time'], -sim['theta'])
        n_rejected = 0
        for i,a in enumerate(sim_trans['age'].values):
            ax.axvline(-a, color = 'k', lw = 0.5)
            if 'GS' in sim_trans['start of'].values[i]:
                ax.axvspan(-a, -a+sim_trans['duration'].values[i],
                           color = 'slategray')

        w_pos, GI_sim_mean, GS_sim_mean, GI_sim_freq, GS_sim_freq = compute_rw_stats(
            sim_trans['age'], sim_trans['start of'], sim_trans['duration'])

        if GI_sim_mean is np.nan:
            print('trajectory %i exhibits a climate period > 20kyr and is omitted' % (ntraj))
            n_rejected +=1
            continue

        ### save data to results array ###

        results[n, : , 0] = GI_sim_mean
        results[n, : , 1] = GS_sim_mean
        results[n, : , 2] = GI_sim_freq
        results[n, : , 3] = GS_sim_freq


    GI_sim_mean_stats = np.percentile(results[:,:,0], [5, 25, 50, 75, 95], axis=0)
    GS_sim_mean_stats = np.percentile(results[:,:,1], [5, 25, 50, 75, 95], axis=0)
    GI_sim_freq_stats = np.percentile(results[:,:,2], [5, 25, 50, 75, 95], axis=0)
    GS_sim_freq_stats = np.percentile(results[:,:,3], [5, 25, 50, 75, 95], axis=0)

    columns = ['age', 'GI_sim_dur_05', 'GI_sim_dur_25',
               'GI_sim_dur_50', 'GI_sim_dur_75', 'GI_sim_dur_95',
               'GS_sim_dur_05', 'GS_sim_dur_25', 'GS_sim_dur_50',
               'GS_sim_dur_75', 'GS_sim_dur_95',
               'GI_sim_freq_05', 'GI_sim_freq_25',
               'GI_sim_freq_50', 'GI_sim_freq_75',
               'GI_sim_freq_95',
               'GS_sim_freq_05', 'GS_sim_freq_25',
               'GS_sim_freq_50', 'GS_sim_freq_75', 'GS_sim_freq_95']

    stats = pd.DataFrame(np.concatenate((w_pos[None, :],
                                         GI_sim_mean_stats,
                                         GS_sim_mean_stats,
                                         GI_sim_freq_stats,
                                         GS_sim_freq_stats), axis=0).T,
                         columns=columns)

    stats.to_csv(dirname(params) + '/duration_stats_%i.csv' %
                 (1000-n_rejected), index=False)

    print(len(stat_list) - n_rejected, 'transient trajectories used in the stastitical evaluation for figure 6')
    return stats

