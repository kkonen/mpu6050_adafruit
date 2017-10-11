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


def dtw_test():

    input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)
    filtered = filter_device(dev_data(input_data, 0))
    segmented = segment_data(filtered, 10000, 'gy')
    mean_seg = calc_mean_segment(segmented)

    for segment in segmented:
       # plot_segment(segment)
        distance, path = fastdtw(segment['az'], mean_seg['az'], dist=euclidean)
        print(distance/100000)

    rnd = np.random.rand(300)*32000
    distance, path = fastdtw(rnd, mean_seg['az'], dist=euclidean)
    print("rnd dist:" + str(distance/100000))
    rnd_d = copy.deepcopy(mean_seg)
    rnd_d['az'] = rnd
    plot_segment(rnd_d, title='rnd')

    z = np.zeros(300)
    distance, path = fastdtw(z, mean_seg['az'], dist=euclidean)
    print("z dist:" + str(distance/100000))
    z_d = copy.deepcopy(mean_seg)
    z_d['az'] = z
    plot_segment(z_d, title='zeros')


    amp_dec = mean_seg['az'] * 0.5
    distance, path = fastdtw(amp_dec, mean_seg['az'], dist=euclidean)
    print("amp_dec dist:" + str(distance/100000))
    amp_dec_d = copy.deepcopy(mean_seg)
    amp_dec_d['az'] = amp_dec
    plot_segment(amp_dec_d, title='amp_dec')


    amp_inc = mean_seg['az'] * 1.5
    distance, path = fastdtw(amp_inc, mean_seg['az'], dist=euclidean)
    print("amp_inc dist:" +str(distance/100000))
    amp_inc_d = copy.deepcopy(mean_seg)
    amp_inc_d['az'] = amp_inc
    plot_segment(amp_inc_d, title='amp_inc')

    y_shift = mean_seg['az'] + 5000
    distance, path = fastdtw(y_shift, mean_seg['az'], dist=euclidean)
    print("y shift dist:" + str(distance/100000))
    y_shift_d = copy.deepcopy(mean_seg)
    y_shift_d['az'] = y_shift
    plot_segment(y_shift_d, title='yshift')

    x_shift = np.append(np.zeros(50), mean_seg['az'][0:250])
    distance, path = fastdtw(x_shift, mean_seg['az'], dist=euclidean)
    print("x shift dist:" + str(distance/100000))
    x_shift_d = copy.deepcopy(mean_seg)
    x_shift_d['az'] = x_shift
    plot_segment(x_shift_d, title='xshift')

    plot_segment(mean_seg, title='Mean Trick')

    plt.show()


dtw_test()

