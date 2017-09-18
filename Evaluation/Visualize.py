import matplotlib.pyplot as plt
import numpy as np

num_dev = 3

data = np.genfromtxt('data/test0.csv', delimiter=',', skip_header=0, names=True)

fig, ax = plt.subplots(num_dev, sharex=True)

for i in range(num_dev):

   # ax[i].set_title(str(i))
   # ax[i].set_xlabel('Time')
   # ax[i].set_ylabel('Acc')

    dev = data[np.where(data[:]['device'] == i)]
    time = dev['micros']

    ax[i].plot(time, dev['ax'], c='r', label='ax')
    ax[i].plot(time, dev['ay'], c='g', label='ay')
    ax[i].plot(time, dev['az'], c='b', label='az')
    ax[i].plot(time, dev['gx'], c='m', label='gx')
    ax[i].plot(time, dev['gy'], c='c', label='gy')
    ax[i].plot(time, dev['gz'], c='k', label='gz')

    leg = ax[i].legend()

plt.show()
