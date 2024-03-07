import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir, exists
from scipy.signal import filtfilt, cheby1, lombscargle, freqz
from scipy.signal import filtfilt, cheby1, lombscargle, freqz
from scipy.interpolate import interp1d, UnivariateSpline


def dirname(params, sub='results'):

    basename = [p + '_%0.2f' % (v)
                for p, v in zip(params.keys(),
                                params.values())]
    basename = basename[:11]
    basename.sort()
    basename = '-'.join(basename)
    if not isdir('output/' + basename):
        mkdir('output/' + basename)

    if not isdir('output/' + basename + '/' + sub):
        mkdir('output/' + basename + '/' + sub)

    return 'output/' + basename + '/' + sub


def check_results(params, filename, recompute=False):
    basename = [p + '_%0.2f' % (v)
                for p, v in zip(params.keys(),
                                params.values())]
    basename = basename[:11]
    basename.sort()
    basename = '-'.join(basename)
    filename = 'output/' + basename + '/results/' + filename

    if exists(filename) and not recompute:
        print(filename + ' exists already')
        return filename, True

    if not isdir('output/' + basename):
        mkdir('output/' + basename)
        print('created results/' + basename)

    # if not isdir('results/' + basename + directory):
    #     mkdir('results/' + basename + directory)
    #     print('created results/' + basename + directory)
    return filename, False


def cheby_lowpass(cutoff, fs, order, rp):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    return b, a


def cheby_lowpass_filter(x, cutoff, fs, order=8, rp=0.05, show_gain=False):

    b, a = cheby_lowpass(cutoff, fs, order, rp)

    y = filtfilt(b, a, x)

    if show_gain:
        wmax = 10 * cutoff * 2 * np.pi / fs
        wmin = 1/100 * cutoff * 2 * np.pi / fs
        ww = np.arange(wmin, wmax, wmin)
        w, h = freqz(b, a, worN=ww)
        fig, ax = plt.subplots()
        ax.plot(w / (2*np.pi) * fs, np.abs(h))
        ax.axvline(cutoff, color='C2', ls=':')
    return y


def preprocess_LR04(t_i, t_f, dt, cutoff):
    ### read LR04 data ###

    LR04 = pd.read_table('proxy_data/Global_stack_d18O.tab',
                         header=57,
                         names=['age', 'd18o', 'd18o std'])

    # linear interpolate to 0.1y resolution

    t_filt = -200000
    LR04_time = -LR04['age'].values * 1000
    LR04_d18o = LR04['d18o'].values
    LR04_interp = UnivariateSpline(LR04_time[::-1], LR04_d18o[::-1], s=0)
    filt_time = np.arange(t_filt, -10000, dt)
    LR04_d18o_01y = LR04_interp(filt_time)

    ### filter interpolated time series with a 10ky lowpass ###
    LR04_d18o_filt = cheby_lowpass_filter(LR04_d18o_01y,
                                          1/cutoff,
                                          1/dt,
                                          order=2,
                                          rp=0.1,
                                          show_gain=True)

    # LR04_d18o_test = cheby_lowpass_filter(LR04_d18o_01y,
    #                                       1/10000,
    #                                       1/dt,
    #                                       order=2,
    #                                       rp=0.1,
    #                                       show_gain=True)

    # crop the filtered LR04 to time period of interest

    mask = (filt_time >= t_i) & (filt_time < t_f)
    LR04_d18o_cropped = LR04_d18o_filt[mask]

    ### normalize cropped data ###
    LR04_d18o_cropped = ((LR04_d18o_cropped - np.mean(LR04_d18o_cropped))
                         / np.std(LR04_d18o_cropped))
    return LR04_d18o_cropped
