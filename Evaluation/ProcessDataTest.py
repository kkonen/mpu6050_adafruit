from FilterAndSegment import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def plot_test():
    input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)

    unfiltered = dev_data(input_data, 0)
    filtered = filter_device(unfiltered)
    segmented = segment_data(filtered, 10000, 'gy')

    plot_segment(filtered, unfiltered, title='Dataset')

    i = 0
    for segment in segmented:
        i += 1
        plot_segment(segment, title='Trick ' + str(i))

    plot_segment(calc_mean_segment(segmented), title='Mean Trick')

    plt.show()


def save_load_test():
    input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)
    filtered = filter_device(dev_data(input_data, 0))
    segmented = segment_data(filtered, 10000, 'gy')

    np.save('ollies_0', segmented)


def dtw_test():

    input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)
    filtered = filter_device(dev_data(input_data, 0))
    segmented = segment_data(filtered, 10000, 'gy')
    mean_seg = calc_mean_segment(segmented)

    test_channel = 'gz'
    t = mean_seg['micros']/1000000

    # compare each sample to the mean sample
    for idx, segment in enumerate(segmented):
        plot_segment(segment, title='Trick #'+str(idx), t=t)
        distance, path = fastdtw(segment[test_channel], mean_seg[test_channel])
        print(distance/100000)

    # compare with random data
    rnd = np.random.rand(300)*32000
    distance, path = fastdtw(rnd, mean_seg[test_channel], dist=euclidean)
    print("rnd dist:" + str(distance/100000))
    # plot_segment(rnd, title='rnd', include_labels=[test_channel], t=t)

    # compare with zero data
    z = np.zeros(300)
    distance, path = fastdtw(z, mean_seg[test_channel], dist=euclidean)
    print("z dist:" + str(distance/100000))
    # plot_segment(z, title='zeros', include_labels=[test_channel], t=t)

    # compare with same sample of decreased amplitude (signal*0.5)
    amp_dec = mean_seg[test_channel] * 0.5
    distance, path = fastdtw(amp_dec, mean_seg[test_channel], dist=euclidean)
    print("amp_dec dist:" + str(distance/100000))
    # plot_segment(amp_dec, title='amp_dec', include_labels=[test_channel], t=t)

    # compare with same sample of increased amplitude (signal*1.5)
    amp_inc = mean_seg[test_channel] * 1.5
    distance, path = fastdtw(amp_inc, mean_seg[test_channel], dist=euclidean)
    print("amp_inc dist:" + str(distance/100000))
    # plot_segment(amp_inc, title='amp_inc', include_labels=[test_channel], t=t)

    # compare with same sample of y-shifted amplitude (signal+5000)
    y_shift = mean_seg[test_channel] + 5000
    distance, path = fastdtw(y_shift, mean_seg[test_channel], dist=euclidean)
    print("y shift dist:" + str(distance/100000))
    # plot_segment(y_shift, title='y-shift', include_labels=[test_channel], t=t)

    # compare with same sample of phase-shifted amplitude (signal shifted right by 50 samples)
    x_shift = np.append(np.zeros(50), mean_seg[test_channel][0:250])
    distance, path = fastdtw(x_shift, mean_seg[test_channel], dist=euclidean)
    print("x shift dist:" + str(distance/100000))
    # plot_segment(x_shift, title='x-shift', include_labels=[test_channel], t=t)

    # plot mean sample
    plot_segment(mean_seg, title='Mean Trick', include_labels=[test_channel])

    plt.show()


dtw_test()
#save_load_test()
