import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz

num_dev = 3

input_data = np.genfromtxt('data/multi_sen_test/12.csv', delimiter=',', skip_header=0, names=True)


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

    return out, t


def filterAndPlotSingleDev(dev_idx):
    fig, ax = plt.subplots(4, sharex=True)

    dev = input_data[np.where(input_data[:]['device'] == dev_idx)]
    time = dev['micros']

    # Filter the data, and plot both the original and filtered signals.
    a_x, t = filterData(dev['ax'], 15, time)
    a_y, t = filterData(dev['ay'], 15, time)
    a_z, t = filterData(dev['az'], 15, time)

    ax[0].plot(t, a_x, c='r', label='ax')
    ax[1].plot(t, a_y, c='g', label='ay')
    ax[2].plot(t, a_z, c='b', label='az')
    ax[3].plot(time/1000000, dev['gx'], c='m', label='gx')
    ax[3].plot(time/1000000, dev['gy'], c='c', label='gy')
    ax[3].plot(time/1000000, dev['gz'], c='k', label='gz')
    plt.show()


def filterAndPlotAllDevs():

    fig, ax = plt.subplots(num_dev, sharex=True)

    for i in range(num_dev):

        # ax[i].set_title(str(i))
        # ax[i].set_xlabel('Time')
        # ax[i].set_ylabel('Acc')
        dev = input_data[np.where(input_data[:]['device'] == i)]
        time = dev['micros']

        a_x, t = filterData(dev['ax'], 15, time)
        a_y, t = filterData(dev['ay'], 15, time)
        a_z, t = filterData(dev['az'], 15, time)

        ax[i].plot(t, a_x, c='r', label='ax')
        ax[i].plot(t, a_y, c='g', label='ay')
        ax[i].plot(t, a_z, c='b', label='az')
        ax[i].plot(time/1000000, dev['gx'], c='m', label='gx')
        ax[i].plot(time/1000000, dev['gy'], c='c', label='gy')
        ax[i].plot(time/1000000, dev['gz'], c='k', label='gz')

        leg = ax[i].legend()

    plt.show()

filterAndPlotAllDevs()