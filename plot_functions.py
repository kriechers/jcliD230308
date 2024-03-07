import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch
from scipy.signal import cheby1, filtfilt, freqz
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm



def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def vmarker(x0, x1, ax0, ax1, **kwargs):
    xy0 = (x0, ax0.get_ylim()[0])
    xy1 = (x1, ax1.get_ylim()[1])
    ax0.axvline(x0, **kwargs)
    ax1.axvline(x1, **kwargs)
    con = ConnectionPatch(xy0, xy1, 'data', 'data',
                          ax0, ax1, **kwargs)
    ax0.add_artist(con)

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(ax, x, y, z=None, cmap=plt.get_cmap('cool'), norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0, zorder = 1):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(list(segments), array=z, cmap=cmap,
                        norm=norm, linewidth=linewidth, alpha=alpha,
                        zorder = zorder)
    #_,ax = plt.subplots()
    ax.add_collection(lc)

    return lc



def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order = 8, rp = 0.05, show_gain =False):

    b, a = cheby_lowpass(cutoff, fs, order, rp)
    
    y = filtfilt(b, a, x)

    if show_gain:
        wmax = 10 * cutoff * 2 * np.pi / fs
        wmin = 1/100 * cutoff * 2 * np.pi /fs
        ww = np.arange(wmin, wmax, wmin)
        w, h = freqz(b,a, worN = ww)
        fig, ax = plt.subplots()
        ax.plot(w / (2*np.pi) * fs, np.abs(h))
        ax.axvline(cutoff, color = 'C2', ls = ':')
    return y


def binning(t_ax, data, bins=None):
    '''
    INPUT
    -----
    data:= data of a time series
    t_ax:= time axis of a time series
    bins:= array like, equally spaced. 
        centers of bins will define new t_axis


    Output

    binned_data:= the i-th entry of binned_data is the 
        the mean of all data points which are located 
        between the i-th and i+1th elements of bins on the 
        t_ax. 
    binned_t_ax := center value of the bins
    '''

    if bins is None:
        res = np.max(np.diff(t_ax))
        bins = np.arange(t_ax[0]-res // 2,
                         t_ax[-1]+res,
                         res)

    binned_data = np.array([np.mean(data[(t_ax > bins[i]) &
                                         (t_ax < bins[i+1])], axis = 0)
                            for i in range(len(bins)-1)])

    binned_t_ax = bins[:-1]+0.5*(bins[1]-bins[0])

    return binned_t_ax, np.array(binned_data)


def NGRIP_stadial_mask(age):
    '''
    this function takes an age axis as an input, that is compatibles 
    with the NGRIP GICC05modelext chronology in terms of covered time. 

    it returen a mask of the same length as the input age axis, which 
    is true, where the age of input corresponds a Greenland stadial. 
    '''

    # load dataset on GI onsets and GS onsets
    stratigraphic = pd.read_excel('../data/Rasmussen_et_al_2014'
                                  + '_QSR_Table_2.xlsx',
                                  header=None,
                                  skiprows=range(23),
                                  names=['event', 'age'],
                                  usecols='A,C')

    # create a mask from the above data which is True for all
    # GI events
    stadials = np.array(['GS' in x for x in stratigraphic['event']])

    # from that mask, a mask is derived that selects only transitions
    # between GI and GS from the stratigraphic data set (the set includes
    # minor GI events that follow major events => GI to GI to GS for example)
    # if transitions[i] is True, there is a transition between the i-th and the
    # i+1-th event from the stratigraphic data set. Since time goes in the
    # opposite direction than age, the age corresponding to the i-th event
    # is the correct age of the transition (that is the point in time where
    # the preceeding phase ended).

    transitions = np.append(np.array(stadials[:-1]) != np.array(stadials[1:]),
                            False)
    transition_ages = stratigraphic['age'][transitions].values

    max_age = np.max(age)
    min_age = np.min(age)

    start_idx = 0
    while transition_ages[start_idx] < min_age:
        start_idx += 1

    end_idx = len(transition_ages)-1
    while transition_ages[end_idx] > max_age:
        end_idx -= 1

    if stadials[start_idx]:
        GS = age < transition_ages[start_idx]

        for i in range(start_idx + 1, end_idx, 2):
            GS_mask = ((transition_ages[i] < age)
                       & (age < transition_ages[i+1]))
            GS = GS | GS_mask
    else:
        GS = np.full(len(age), False)
        for i in range(start_idx, end_idx, 2):
            GS_mask = ((transition_ages[i] < age)
                       & (age < transition_ages[i+1]))
            GS = GS | GS_mask

    return GS, transition_ages[start_idx: end_idx+1]
