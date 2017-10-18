from FilterAndSegment import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.lib.recfunctions import append_fields, drop_fields

def plot():
    input_data = np.genfromtxt('data/max_sens_tricks/new_pop_shuv_it.csv', delimiter=',', skip_header=0, names=True)
    #input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)

    unfiltered = dev_data(input_data, 0)
    filtered = filter_device(unfiltered)
    segmented = segment_data(filtered, 'gy')

    plot_segment(filtered, unfiltered, title='Dataset')

    i = 0
    for segment in segmented:
        i += 1
        plot_segment(segment, title='Trick ' + str(i))

    plot_segment(calc_mean_segment(segmented), title='Mean Trick')

    plt.show()


def save_load(trick_names, prefix):
    device = 2
    for trick_name in trick_names:
        input_data = np.genfromtxt(prefix+trick_name+'.csv', delimiter=',', skip_header=0, names=True)
        filtered = filter_device(dev_data(input_data, device))

        segmented = segment_data(filtered, 'gy')

        np.save(trick_name+'_dev'+str(device), segmented)


def pre_process_and_save(trick_names, prefix_load='data/max_sens_tricks/new_', prefix_save='data/processed_data/'):

    devices = [0, 2]
    filtered = dict()

    for trick_name in trick_names:
        for dev in devices:
            input_data = np.genfromtxt(prefix_load+trick_name+'.csv', delimiter=',', skip_header=0, names=True)
            filtered[dev] = filter_device(dev_data(input_data, dev))
            #segmented = segment_data(filtered[dev], 'gy')
        joined = join_data(filtered)
        segmented = segment_data(joined, 'gy')

        print(segmented)
        np.save(prefix_save + trick_name, segmented)


def join_data(input_data):
    i = 0
    for dev in input_data.keys():
        if i == 0:
            joined = input_data[dev]
        else:
            for chan in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                joined = append_fields(joined, chan+str(dev), input_data[dev][chan])
            joined = drop_fields(joined, 'device')
        i += 1

    print(joined.dtype.names)
    return joined


def calc_dtw_sum(seg1, seg2, channels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], weights=[1, 1, 1, 1, 1, 1]):
    #channels = ['gx', 'gy', 'gz']
    #weights = [1, 1, 1]
    #channels = ['gx', 'gy', 'gz', 'gx2', 'gy2', 'gz2']
    #weights = [1, 1, 1, 1, 1, 1]

    dtw_sum = 0
    for i in range(0, len(channels)):
        dist, path = fastdtw(seg1[channels[i]], seg2[channels[i]])
        dtw_sum += dist*weights[i]

    return (dtw_sum/sum(weights))/100000


def dtw():
    input_data = np.genfromtxt('data/multi_sen_test/11.csv', delimiter=',', skip_header=0, names=True)
    filtered = filter_device(dev_data(input_data, 0))
    segmented = segment_data(filtered, 'gy')
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


def train():
    trick_names = ['ollie', 'nollie', 'pop_shuv_it']
    trick = dict()
    mean_trick = dict()

    for trick_name in trick_names:
        i = 0
        trick[trick_name] = np.load('data/processed_data/' + trick_name+'.npy')
        mean_trick[trick_name] = calc_mean_segment(trick[trick_name])
        plot_segment(mean_trick[trick_name], title='mean '+trick_name)
        for t in trick[trick_name]:
            i += 1
            #print(trick_name + '_' + str(i) + ' dist to own mean: ' + str(calc_dtw_sum(t, mean_trick[trick_name])))

    channels = mean_trick[trick_names[0]].dtype.names
    channels = channels[1:]  # remove micros
    weights = np.ones(len(channels))
    for trick_name in trick_names:
        i = 0
        for t in trick[trick_name]:
            i += 1
            dist = dict()
            d_str = ''
            min_d = 99999999999999999999
            best_lb = ''
            for comp in trick_names:
                dist[comp] = calc_dtw_sum(t, mean_trick[comp], channels, weights)
                d_str = d_str + '[' + comp + ':' + str(dist[comp]) + ']'
                if dist[comp] < min_d:
                    best_lb = comp
                    min_d = dist[comp]

            correct = '1 # ' if best_lb == trick_name else '0 # '
            print(correct + trick_name + '_' + str(i) + ' [class: ' + best_lb + '] - ' + d_str)

    #plt.show()


def combine_devices():
    devices = [0, 2]
    trick_names = ['ollie', 'nollie', 'pop_shuv_it']

    trick = dict()
    mean_trick = dict()
    combined_mean = dict()

    for dev in devices:
        for trick_name in trick_names:
            tr = trick_name+'_dev'+str(dev)
            trick[tr] = np.load(tr+'.npy')
            mean_trick[tr] = calc_mean_segment(trick[tr])

    for trick_name in trick_names:
        combined_mean[trick_name] = mean_trick[trick_name+'_dev'+str(devices[0])]
        for i in range(1, len(devices)):
            sd = str(devices[i])
            tr = trick_name+'_dev' + sd
            for chan in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
                combined_mean[trick_name] = append_fields(combined_mean[trick_name],
                                                          chan+sd,
                                                          mean_trick[tr][chan])
        combined_mean[trick_name] = drop_fields(combined_mean[trick_name], 'device')
        print(combined_mean[trick_name].dtype.names)



#plot()
#train()
#combine_devices()
#save_load()

#pre_process_and_save(['ollie', 'nollie', 'pop_shuv_it'])
train()