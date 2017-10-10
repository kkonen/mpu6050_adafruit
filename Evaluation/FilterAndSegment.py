import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
import copy

num_dev = 3

input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def filterData(data, cutoff, time, order=6):
    # Filter requirements.
    fs = time.size / (time[-1]/1000000)        # sample rate, Hz
    T = (time[-1]/1000000)      # seconds
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    out = butter_lowpass_filter(data, cutoff, fs, order)

    return out


def devData(input, idx):
    return input[np.where(input[:]['device'] == idx)]


def filterDevice(input):
   # if input.dtype == np.dtype('float64'):
    time = input['micros']
    output = copy.deepcopy(input)
    freq = 10
    output['ax'] = filterData(input['ax'], freq, time)
    output['ay'] = filterData(input['ay'], freq, time)
    output['az'] = filterData(input['az'], freq, time)
    output['gx'] = filterData(input['gx'], freq, time)
    output['gy'] = filterData(input['gy'], freq, time)
    output['gz'] = filterData(input['gz'], freq, time)
    return output


def findPeaks(signal, ts, thresh, min_dist):
    peaks = []
    lastPeakMicros = 0
    for i in range(0,len(signal)):
        if (signal[i] > thresh) and ts[i] - lastPeakMicros > min_dist:
            lastPeakMicros = ts[i]
            peaks.append(i)

    return np.array(peaks)


def segmentData(input, threshold, channel):
    peaks = findPeaks(input[:][channel], input[:]['micros'], threshold, 1000000)
    segments = []
    for i in peaks:
        segments.append(input[i-50:i+250])

    return np.array(segments)


def meanSegment(segments):
    mean_seg = copy.deepcopy(segments[0])
    mean_seg['ax'] = segments['ax'].mean(axis=0)
    mean_seg['ay'] = segments['ay'].mean(axis=0)
    mean_seg['az'] = segments['az'].mean(axis=0)
    mean_seg['gx'] = segments['gx'].mean(axis=0)
    mean_seg['gy'] = segments['gy'].mean(axis=0)
    mean_seg['gz'] = segments['gz'].mean(axis=0)

    return mean_seg


def plotSegment(input, input2=None, title=None):
    if input2 is None:
        fig, ax = plt.subplots(1)
        t = input['micros']/1000000
        ax.plot(t, input['ax'], c='r', label='ax')
        ax.plot(t, input['ay'], c='g', label='ay')
        ax.plot(t, input['az'], c='b', label='az')
        ax.plot(t, input['gx'], c='m', label='gx')
        ax.plot(t, input['gy'], c='c', label='gy')
        ax.plot(t, input['gz'], c='k', label='gz')
    else:
        fig, ax = plt.subplots(2, sharex=True)
        t = input['micros']/1000000
        ax[0].plot(t, input['ax'], c='r', label='ax')
        ax[0].plot(t, input['ay'], c='g', label='ay')
        ax[0].plot(t, input['az'], c='b', label='az')
        ax[0].plot(t, input['gx'], c='m', label='gx')
        ax[0].plot(t, input['gy'], c='c', label='gy')
        ax[0].plot(t, input['gz'], c='k', label='gz')

        ax[1].plot(t, input2['ax'], c='r', label='ax')
        ax[1].plot(t, input2['ay'], c='g', label='ay')
        ax[1].plot(t, input2['az'], c='b', label='az')
        ax[1].plot(t, input2['gx'], c='m', label='gx')
        ax[1].plot(t, input2['gy'], c='c', label='gy')
        ax[1].plot(t, input2['gz'], c='k', label='gz')
    if title is not None:
        fig.canvas.set_window_title(title)


unfiltered = devData(input_data, 0)
filtered = filterDevice(unfiltered)
segmented = segmentData(filtered, 10000, 'gy')

plotSegment(filtered, unfiltered, title='Dataset')

i = 0
for segment in segmented:
    i = i+1
    plotSegment(segment, title='Trick '+str(i))

plotSegment(meanSegment(segmented), title='Mean Trick')

plt.show()