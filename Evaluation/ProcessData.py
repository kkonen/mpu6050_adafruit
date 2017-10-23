from FilterAndSegment import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.lib.recfunctions import append_fields, drop_fields
from os import listdir
from os.path import isfile, join

def plot():
    input_data = np.genfromtxt('data/max_sens_tricks/new_popshuvit.csv', delimiter=',', skip_header=0, names=True)
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


def load_from_dir_and_preprocess(trick_name, prefix_save):
    devices = [0, 2]
    onlyfiles = [f for f in listdir('data/tricks/' + trick_name) if isfile(join('data/tricks/' + trick_name, f))]
    print(onlyfiles)
    segmented = []
    numsum = 0
    for file in onlyfiles:
        filtered = dict()
        print(file)
        input_data = np.genfromtxt( 'data/tricks/'+trick_name+'/'+file, delimiter=',', skip_header=0, names=True)
        for dev in devices:
            filtered[dev] = filter_device(dev_data(input_data, dev))
        joined = join_data(filtered)
        segments = segment_data(joined, 'gy')
        num =len(segments)
        #numsum += num
        #print(num)
        if num == 2:
           # for seg in segments:
               # plot_segment(seg,title=file, include_labels=['gy', 'gy2'])


            segments = segments[:-1]
         #   segments = segments[1:]

            #plot_segment(segments[0], include_labels=['gz','gz2'])
            #print('tossed segment, new len: '+str(len(segments)))

            print(segments.dtype.names)

        #else:
        #    continue

        if len(segments) != 1:
            print('NOT EXACTLY 1 TRICK: '+str(num))
            continue
        segmented.append(segments[0])
        print('append:')
        print(np.array(segmented).dtype.names)
    #print(numsum)
    #plt.show()
    np.save(prefix_save+trick_name, segmented)

def pre_process_and_save(trick_names, prefix_load='data/max_sens_tricks/new_', prefix_save='data/processed_data/old/'):

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

   # print(joined.dtype.names)
    return joined


def calc_dtw_sum(seg1, seg2, channels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], weights=None):

    if weights is None:
        weights = np.ones(len(channels))

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


def train(save_means=False):
    trick_names = ['ollie', 'nollie', 'popshuvit']
    trick = dict()
    mean_trick = dict()

    for trick_name in trick_names:
        trick[trick_name] = np.load('data/processed_data/' + trick_name+'.npy')
        mean_trick[trick_name] = calc_mean_segment(trick[trick_name])
        if save_means:
            np.save('data/processed_data/means/mean_'+trick_name, np.array(mean_trick[trick_name]))
        plot_segment(mean_trick[trick_name], title='mean '+trick_name)


def tst(path, mean_path):
    from termcolor import colored
    thresh = 2.5

    tst_data = [f for f in listdir(path) if isfile(join(path, f))]
    mean_data = [f for f in listdir(mean_path) if isfile(join(mean_path, f))]
    mean_data = np.array(mean_data)[np.chararray.startswith(mean_data, 'mean_')]
    print(mean_data)
    print(tst_data)

    #trick_names = ['ollie', 'nollie', 'popshuvit']
    test_tricks = [tr.replace('.npy', '') for tr in tst_data]
    learned_tricks = [tr.replace('.npy', '').replace('mean_', '') for tr in mean_data]
    trick = dict()
    mean_trick = dict()

    for trick_name in test_tricks:
        trick[trick_name] = np.load(path + trick_name+'.npy')
        print('found '+str(len(trick[trick_name])) + ' '+trick_name+'s')

    for trick_name in learned_tricks:
        mean_trick[trick_name] = np.load(mean_path + 'mean_' + trick_name+'.npy')

    channels = mean_trick[learned_tricks[0]].dtype.names
    channels = channels[1:]  # remove micros
    weights = np.ones(len(channels))
    for trick_name in test_tricks:
        i = 0
        for t in trick[trick_name]:
            i += 1
            dist = dict()
            d_str = ''
            min_d = float('inf')
            best_lb = ''
            for comp in learned_tricks:
                dist[comp] = round(calc_dtw_sum(t, mean_trick[comp], channels, weights), 3)
                d_str = d_str + '[' + comp + ':' + str(dist[comp]) + ']'
                if dist[comp] < min_d:
                    best_lb = comp
                    min_d = dist[comp]

            if min_d > thresh:
                best_lb = 'rejected'
                d_str = d_str.replace(str(min_d), colored(str(min_d), 'red'))
            else:
                d_str = d_str.replace(str(min_d), colored(str(min_d), 'blue'))

            best_lb = colored(best_lb, 'green') if best_lb == trick_name else colored(best_lb, 'red')
            print(trick_name + '_' + str(i) + ' [class: ' + best_lb + '] - ' + d_str)


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

def merge_mis():
    n = np.load('data/processed_data/mis/rejected_n.npy')
    o = np.load('data/processed_data/mis/rejected_o.npy')
    psi = np.load('data/processed_data/mis/rejected_psi.npy')
    rejected = n
    rejected = np.append(rejected, np.array([o[0]]), axis=0)
    #rejected = np.append(rejected, np.array([o[1]]), axis=0)
    rejected = np.append(rejected, psi, axis=0)
    np.save('data/processed_data/mis/rejected', rejected)

#plot()
#train()
#combine_devices()
#save_load()

#pre_process_and_save(['ollie', 'nollie', 'popshuvit'])


#load_from_dir_and_preprocess('popshuvit', 'data/processed_data/')
#load_from_dir_and_preprocess('nollie', 'data/processed_data/')
#load_from_dir_and_preprocess('ollie', 'data/processed_data/')
#train(True)


#load_from_dir_and_preprocess('popshuvit', 'data/processed_data/mis/')
#load_from_dir_and_preprocess('nollie', 'data/processed_data/mis/')
#load_from_dir_and_preprocess('ollie', 'data/processed_data/mis/')



print('old')
tst('data/processed_data/old/', 'data/processed_data/means/')
print('new')
tst('data/processed_data/', 'data/processed_data/means/')
#print('mis')
#tst('data/processed_data/mis/', 'data/processed_data/means/')