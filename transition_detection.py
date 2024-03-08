import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from miscellanious import dirname
from miscellanious import check_results, dirname


def detect_transitions_sim(sim, params):

    # load ice bifurcation points
    directory = dirname(params)
    ice_bifurcations = pd.read_csv(directory + '/ice_bifurcations.csv')
    Ic = ice_bifurcations['I'][0]

    #################################################################
    # detect transition in simulation                               #
    #################################################################

    warm_idx = np.argwhere(np.diff(sim['I'] < Ic)).flatten()
    cool_idx = np.argwhere(np.diff(sim['I'] > 0.5)).flatten()
    trans_idx = np.sort(np.append(warm_idx, cool_idx))
    DO_event_list = [sim['time'][0]]
    DO_cooling_list = [sim['time'][0]]

    if sim['I'][0] > 0.5:
        GS = True
        stadial_start = True
    else:
        GS = False
        stadial_start = False

    for t in trans_idx:
        if (GS and all(sim['I'][t-5:t] > Ic)
            and t in warm_idx
                and (all(sim['I'][t+1:t+4] < Ic))):
            DO_event_list.append(sim['time'][t])
            GS = False
        elif not GS and (np.mean(sim['I'][t:t+5]) > 0.5) and t in cool_idx:
            DO_cooling_list.append(sim['time'][t])
            GS = True

    if stadial_start:
        DO_event_list.pop(0)
        if len(DO_cooling_list) > len(DO_event_list):
            DO_event_list.append(sim['time'].iloc[-1])
            stadial_end = True
        else:
            DO_cooling_list.append(sim['time'].iloc[-1])
            stadial_end = False

    if not stadial_start:
        DO_cooling_list.pop(0)
        if len(DO_event_list) > len(DO_cooling_list):
            DO_cooling_list.append(sim['time'].iloc[-1])
            stadial_end = False
        else:
            DO_event_list.append(sim['time'].iloc[-1])
            stadial_end = True

    #print(np.array(DO_cooling_list[:-1]) - np.array(DO_event_list[1:]))
    #print(np.array(DO_event_list) - np.array(DO_cooling_list))
    if sim['time'][0] < 0:
        transitions = np.sort(-np.concatenate((DO_cooling_list,
                                               DO_event_list))).astype(np.int32)
    else:
        transitions = np.sort(np.concatenate((DO_cooling_list,
                                              DO_event_list))).astype(np.int32)
    durations = np.diff(transitions)
    events = []
    i = 0  # in case a time series has no transitions
    if stadial_end:
        for i in range(int(len(durations)/2)):
            events.append('GS-%i' % (i))
            events.append('GI-%i' % (i))
        GS_durations = durations[::2]
        GI_durations = durations[1::2]
        if np.mod(len(durations), 2) == 1:
            events.append('GS-%i' % (i+1))
            GI_durations = np.append(GI_durations, 0)

    else:
        for i in range(int(len(durations)/2)):
            events.append('GI-%i' % (i))
            events.append('GS-%i' % (i))
        GI_durations = durations[::2]
        GS_durations = durations[1::2]
        if np.mod(len(durations), 2) == 1:
            events.append('GI-%i' % (i+1))
            GS_durations = np.append(GS_durations, 0)

    transitions_df = pd.DataFrame(columns=['age', 'start of', 'duration'])
    if sim['time'][0] < 0:
        transitions_df['age'] = transitions[1:]
        transitions_df['duration'] = durations
    else:
        transitions_df['age'] = transitions[:-1][::-1]
        transitions_df['duration'] = durations[::-1]
    transitions_df['start of'] = events

    durations_df = pd.DataFrame(columns=['GI', 'GS'])
    durations_df['GI'] = GI_durations
    durations_df['GS'] = GS_durations

    # fig, ax = plt.subplots()
    # ax.plot(sim['time'], sim['theta'])
    # ax.set_ylim(2, 0.5)
    # for t,s in zip(DO_cooling_list[:-1], DO_event_list[1:]):
    #     ax.axvspan(t,
    #                s,
    #                alpha=0.2,
    #                edgecolor=None,
    #                facecolor='slategray',
    #                zorder=0)

    return transitions_df, durations_df


def transitions_proxy_record():

    stratigraphic = pd.read_excel('proxy_data/Rasmussen_et_al_2014'
                                  + '_QSR_Table_2.xlsx',
                                  header=None,
                                  skiprows=range(23),
                                  names=['event', 'age'],
                                  usecols='A,C')

    stadials = np.array(['GS' in x for x in stratigraphic['event']])
    interstadials = np.array(['GI' in x for x in stratigraphic['event']])
    # from that mask, a mask is derived that selects only transitions
    # between GI and GS from the stratigraphic data set (the set includes
    # minor GI events that follow major events => GI to GI to GS for example)
    # if transitions[i] is True, there is a transition between the i-th and the
    # i+1-th event from the stratigraphic data set. Since time goes in the
    # opposite direction than age, the age corresponding to the i-th event
    # is the correct age of the transition (that is the point in time where
    # the preceeding phase ended)self.

    transitions = np.append(np.array(stadials[:-1]) != np.array(stadials[1:]),
                            False)
    transition_ages = stratigraphic['age'][transitions].values
    events = stratigraphic['event'][transitions].values
    for i, s in enumerate(events):
        if s[-1] in 'abcdefg':
            events[i] = s[:-1]
        events[i] = events[i][9:]
    events[0] = 'Holocence'

    durations = np.append(transition_ages[0], np.diff(transition_ages))

    data = {'GS': durations[1::2],
            'GI': durations[2::2]}

    duration_df = pd.DataFrame(data, columns=['GI', 'GS'], index=[
                               s[3:] for s in events[2::2]])
    complete_df = pd.DataFrame(columns=['age', 'start of', 'duration'])
    complete_df['age'] = transition_ages
    complete_df['start of'] = events
    complete_df['duration'] = durations

    return complete_df, duration_df
