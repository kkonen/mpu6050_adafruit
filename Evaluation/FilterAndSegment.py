import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import copy


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filter_data(data, cutoff, time, order=6):
    # Filter requirements.
    fs = time.size / (time[-1]/1000000)        # sample rate, Hz
    T = (time[-1]/1000000)      # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    out = butter_lowpass_filter(data, cutoff, fs, order)

    return out


def dev_data(input_data, idx):
    return input_data[np.where(input_data[:]['device'] == idx)]


def filter_device(input_data):
   # if input.dtype == np.dtype('float64'):
    time = input_data['micros']
    output = copy.deepcopy(input_data)
    freq = 10
    output['ax'] = filter_data(input_data['ax'], freq, time)
    output['ay'] = filter_data(input_data['ay'], freq, time)
    output['az'] = filter_data(input_data['az'], freq, time)
    output['gx'] = filter_data(input_data['gx'], freq, time)
    output['gy'] = filter_data(input_data['gy'], freq, time)
    output['gz'] = filter_data(input_data['gz'], freq, time)
    return output


def find_peaks(signal, ts, thresh, min_dist):
    peaks = []
    last_peak_micros = 0
    for i in range(0, len(signal)):
        if (signal[i] > thresh) and ts[i] - last_peak_micros > min_dist:
            last_peak_micros = ts[i]
            peaks.append(i)

    return np.array(peaks)


def segment_data(input_data, threshold, channel):
    peaks = find_peaks(input_data[:][channel], input_data[:]['micros'], threshold, 1000000)
    segments = []
    for i in peaks:
        segments.append(input_data[i-50:i+250])

    return np.array(segments)


def calc_mean_segment(segments):
    mean_seg = copy.deepcopy(segments[0])
    mean_seg['ax'] = segments['ax'].mean(axis=0)
    mean_seg['ay'] = segments['ay'].mean(axis=0)
    mean_seg['az'] = segments['az'].mean(axis=0)
    mean_seg['gx'] = segments['gx'].mean(axis=0)
    mean_seg['gy'] = segments['gy'].mean(axis=0)
    mean_seg['gz'] = segments['gz'].mean(axis=0)

    return mean_seg


def plot_segment(input_data, input_data2=None, title=None):
    if input_data2 is None:
        fig, ax = plt.subplots(1)
        t = input_data['micros']/1000000
       # ax.plot(t, input_data['ax'], c='r', label='ax')
      #  ax.plot(t, input_data['ay'], c='g', label='ay')
        ax.plot(t, input_data['az'], c='b', label='az')
       # ax.plot(t, input_data['gx'], c='m', label='gx')
       # ax.plot(t, input_data['gy'], c='c', label='gy')
      #  ax.plot(t, input_data['gz'], c='k', label='gz')
    else:
        fig, ax = plt.subplots(2, sharex=True)
        t = input_data['micros']/1000000
        ax[0].plot(t, input_data['ax'], c='r', label='ax')
        ax[0].plot(t, input_data['ay'], c='g', label='ay')
        ax[0].plot(t, input_data['az'], c='b', label='az')
        ax[0].plot(t, input_data['gx'], c='m', label='gx')
        ax[0].plot(t, input_data['gy'], c='c', label='gy')
        ax[0].plot(t, input_data['gz'], c='k', label='gz')

        ax[1].plot(t, input_data2['ax'], c='r', label='ax')
        ax[1].plot(t, input_data2['ay'], c='g', label='ay')
        ax[1].plot(t, input_data2['az'], c='b', label='az')
        ax[1].plot(t, input_data2['gx'], c='m', label='gx')
        ax[1].plot(t, input_data2['gy'], c='c', label='gy')
        ax[1].plot(t, input_data2['gz'], c='k', label='gz')
    if title is not None:
        fig.canvas.set_window_title(title)

