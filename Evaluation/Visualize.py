import matplotlib.pyplot as plt
import numpy as np

num_dev = 4

data = np.genfromtxt('data/kai_records/realdata03.csv', delimiter=',', skip_header=0, names=True)


dev = data[np.where(data[:]['device'] == 0)]
time = dev['micros']/1000000


fig, ax = plt.subplots(4, sharex=True)

ax[0].plot(time, dev['ax'], c='r', label='ax')
ax[1].plot(time, dev['ay'], c='g', label='ay')
ax[2].plot(time, dev['az'], c='b', label='az')
ax[3].plot(time, dev['gx'], c='m', label='gx')
ax[3].plot(time, dev['gy'], c='c', label='gy')
ax[3].plot(time, dev['gz'], c='k', label='gz')




# fig, ax = plt.subplots(num_dev, sharex=True)

# for i in range(num_dev):
#
#    # ax[i].set_title(str(i))
#    # ax[i].set_xlabel('Time')
#    # ax[i].set_ylabel('Acc')
#
#     dev = data[np.where(data[:]['device'] == i)]
#     time = dev['micros']/1000000



    # ax[i].plot(time, dev['ax'], c='r', label='ax')
    # ax[i].plot(time, dev['ay'], c='g', label='ay')
    # ax[i].plot(time, dev['az'], c='b', label='az')
    # ax[i].plot(time, dev['gx'], c='m', label='gx')
    # ax[i].plot(time, dev['gy'], c='c', label='gy')
    # ax[i].plot(time, dev['gz'], c='k', label='gz')
    #
    # leg = ax[i].legend()

plt.show()
