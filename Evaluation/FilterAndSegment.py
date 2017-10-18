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


def find_peaks(signal, ts, thresh, min_dist, absolute=False):
    peaks = []
    last_peak_micros = 0
    for i in range(0, len(signal)):
        if (signal[i] > thresh or (absolute and signal[i] < -thresh)) and ts[i] - last_peak_micros > min_dist:
            last_peak_micros = ts[i]
            peaks.append(i)

    return np.array(peaks)


def segment_data(input_data, channel, threshold=5000):
    peaks = find_peaks(input_data[:][channel], input_data[:]['micros'], threshold, 1000000, absolute=True)
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


def plot_segment(input_data, input_data2=None, title=None, include_labels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], t=None):
    all_labels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    lb_colors = ['r', 'g', 'b', 'm', 'c', 'k']

    if input_data.dtype == np.dtype('float64'):
        if t is None:
            print("no time given!")
            return
        else:
            fig, ax = plt.subplots(1)
            ax.plot(t, input_data, c='r')
    else:
        if t is None:
            t = input_data['micros']/1000000

        if input_data2 is None:
            fig, ax = plt.subplots(1)
            for i in range(0, len(all_labels)):
                if all_labels[i] in include_labels:
                    ax.plot(t, input_data[all_labels[i]], c=lb_colors[i], label=all_labels[i])
        else:
            fig, ax = plt.subplots(2, sharex=True)
            for i in range(0, len(all_labels)):
                if all_labels[i] in include_labels:
                    ax[0].plot(t, input_data[all_labels[i]], c=lb_colors[i], label=all_labels[i])

            for i in range(0, len(all_labels)):
                if all_labels[i] in include_labels:
                    ax[1].plot(t, input_data2[all_labels[i]], c=lb_colors[i], label=all_labels[i])

    if title is not None:
            fig.canvas.set_window_title(title)
