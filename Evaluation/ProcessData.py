from FilterAndSegment import *
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy.lib.recfunctions import append_fields, drop_fields
from os import listdir
from os.path import isfile, join
from termcolor import colored

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
    devices = [0, 1, 2]
    onlyfiles = [f for f in listdir('data/tricks/' + trick_name) if isfile(join('data/tricks/' + trick_name, f))]
    print(onlyfiles)
    segmented = []
    numsum = 0
    for file in onlyfiles:
        filtered = dict()
        print(file)
        input_data = np.genfromtxt('data/tricks/'+trick_name+'/'+file, delimiter=',', skip_header=0, names=True)
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
    return joined


def calc_dtw_sum(seg1, seg2, channels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], weights=None):

    if weights is None or len(weights) != len(channels):
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


def classify(means, candidate, channels=None, weights=None, rejection_threshold=2.5):
    if channels is None:
        channels = candidate.dtype.names
        channels = channels[1:]  # remove micros

    dist = dict()
    min_d = float('inf')
    best_lb = ''
    for comp in means.keys():
        dist[comp] = round(calc_dtw_sum(candidate, means[comp], channels, weights), 3)
        if dist[comp] < min_d:
            best_lb = comp
            min_d = dist[comp]

    if rejection_threshold is not None and min_d > rejection_threshold:
        best_lb = 'rejected'

    return dist, min_d, best_lb


def tst(path, mean_path):
    thresh = 2.5

    tst_data = [f for f in listdir(path) if isfile(join(path, f))]
    mean_data = [f for f in listdir(mean_path) if isfile(join(mean_path, f))]
    mean_data = np.array(mean_data)[np.chararray.startswith(mean_data, 'mean_')]

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

            dist, min_d, best_lb = classify(mean_trick, t, channels, weights, rejection_threshold=thresh)
            # dist = dict()
            # d_str = ''
            # min_d = float('inf')
            # best_lb = ''
            # for comp in learned_tricks:
            #     dist[comp] = round(calc_dtw_sum(t, mean_trick[comp], channels, weights), 3)
            #     d_str = d_str + '[' + comp + ':' + str(dist[comp]) + ']'
            #     if dist[comp] < min_d:
            #         best_lb = comp
            #         min_d = dist[comp]
            #
            # if min_d > thresh:
            #     best_lb = 'rejected'
            #     d_str = d_str.replace(str(min_d), colored(str(min_d), 'red'))
            # else:
            #     d_str = d_str.replace(str(min_d), colored(str(min_d), 'blue'))

            # if min_d < 0.7:
            #     plot_segment(t, include_labels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], title=trick_name + '_' + str(i), legend=True)
            # elif best_lb != trick_name:
            #     plot_segment(t, include_labels=['ax', 'ay', 'az', 'gx', 'gy', 'gz'], title='mis_'+trick_name + '_' + str(i), legend=True)

            best_lb = colored(best_lb, 'green') if best_lb == trick_name else colored(best_lb, 'red')
            print(trick_name + '_' + str(i) + ' [class: ' + best_lb + '] - ', dist)


def cross_validate(k=5, rejection_threshold=None, channel_runs=None, ignore=['rejected']):
    path = 'data/processed_data/'

    trick_names = [tr.replace('.npy', '') for tr in [f for f in listdir(path) if isfile(join(path, f))]]

    input_trick = dict()
    trick = dict()

    min_num = float('inf')

    for trick_name in trick_names:
        if trick_name not in ignore:
            input_trick[trick_name] = np.load(path + trick_name+'.npy')
            #print('found '+str(len(input_trick[trick_name])) + ' '+trick_name+'s')
            if min_num > len(input_trick[trick_name]):
                min_num = len(input_trick[trick_name])

    # balance the number of tricks
    for trick_name in input_trick.keys():
        np.random.shuffle(input_trick[trick_name])
        trick[trick_name] = copy.deepcopy(input_trick[trick_name][0:min_num])

    # calc split sizes
    import math
    test_len = math.ceil(min_num/k)
    #print('using', min_num, 'samples per class for balancing. (', test_len, 'for testing, ', min_num - test_len, 'for training)')

    acc_array = []
    margin_array = []

    for channels in channel_runs:

        # perform k-fold
        rej_txt = 'without rejection' if rejection_threshold is None \
            else 'with a rejection-threshold of '+str(rejection_threshold)
        print('\nperforming', str(k)+'-fold', 'cross-validation', rej_txt, 'using channels', channels)
        correct = []
        margin = []
        confusion = dict()

        for i in range(k):

            test_data = []
            means = dict()

            # split data in training and test set
            if i == 0:
                tst_set = np.r_[0:test_len]
                tr_set = np.r_[test_len:min_num]
            elif i == k - 1:
                tst_set = np.r_[min_num - test_len:min_num]
                tr_set = np.r_[0:min_num - test_len]
            else:
                tst_set = np.r_[i * test_len:i * test_len + test_len]
                tr_set = np.r_[0:i * test_len, i * test_len + test_len:min_num]
            # print('testidx:', len(tst_set), tst_set, 'tr_idx:', len(tr_set), tr_set)

            for trick_name in trick.keys():
                trick_data = trick[trick_name]

                train_data = trick_data[tr_set]
                means[trick_name] = calc_mean_segment(train_data)

                td = trick_data[tst_set]
                for t in range(0, test_len):
                    test_data.append((td[t-1], trick_name))

                #if i == 0:
                #    plot_segment(means[trick_name],title='mean '+trick_name)




            # run classification

            correct.append(0)
            margin.append(0)
            for candidate in test_data:
                true_label = candidate[1]
                class_result = classify(means, candidate[0],
                                        channels=channels,
                                        rejection_threshold=rejection_threshold)
                classified_label = class_result[2]

                d_str = ''
                for cl in class_result[0]:
                    d_str += '[' + cl + ':' + str(round(class_result[0][cl], 3)) + ']'
                try:
                    confusion[true_label][classified_label] = confusion[true_label][classified_label] + 1
                except KeyError:
                    try:
                        confusion[true_label][classified_label] = 1
                    except KeyError:
                        confusion[true_label]=dict()
                        confusion[true_label][classified_label] = 1

                s_dist = sorted(class_result[0].values())
                marg = s_dist[1]-s_dist[0]
                #print(s_dist, marg)
                margin[i] += marg


                if true_label == classified_label:
                    #print('\tclassified', true_label, 'as', classified_label + '\t\t' + d_str)
                    correct[i] += 1
                else:
                    pass
                    print('\tclassified', true_label, 'as', classified_label + '\t\t' + colored(d_str,'red'))
                    #plot_segment(candidate[0],means[true_label], title="mis: "+candidate[1]+' as '+classified_label)

         #   print('fold', i + 1, 'of', k, ':',
         #         colored(str(round(correct[i]/len(test_data)*100, 2)) + '%', 'blue') + ' correct.',
         #         '(', correct[i], 'of', len(test_data), ')')

        total_acc = 100 * round(sum(correct) / (len(test_data)*k), 2)
        total_margin = round(sum(margin)/(len(test_data)*k), 3)
        acc_array.append(total_acc)
        margin_array.append(total_margin)

        print('\ntotal:', colored(str(total_acc) + '%', 'green',
                                  attrs=['bold']) + ' correct.',
              '(',  sum(correct), 'of', k*len(test_data), ')')

        print('margin: ', total_margin)

        print(confusion, '\n\n')
        plt.show()
    return acc_array, margin_array

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





#load_from_dir_and_preprocess('popshuvit', 'data/processed_data/')
#load_from_dir_and_preprocess('nollie', 'data/processed_data/')
#load_from_dir_and_preprocess('ollie', 'data/processed_data/')
#train(True)


#load_from_dir_and_preprocess('popshuvit', 'data/processed_data/mis/')
#load_from_dir_and_preprocess('nollie', 'data/processed_data/mis/')
#load_from_dir_and_preprocess('ollie', 'data/processed_data/mis/')


#print('\n#######\n# OLD #\n#######')
#tst('data/processed_data/old/', 'data/processed_data/means/')

#print('\n#######\n# NEW #\n#######')
#tst('data/processed_data/', 'data/processed_data/means/')
#plt.show()


chan_runs = [['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
             ['ax', 'ay', 'az', 'gx', 'gy', 'gz',
              'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2'],
             ['ax', 'ay', 'az', 'gx', 'gy', 'gz',
              'ax1', 'ay1', 'az1', 'gx1', 'gy1', 'gz1',
              'ax2', 'ay2', 'az2', 'gx2', 'gy2', 'gz2']]
accuracies = []
margins = []

runs = 10

for i in range(runs):
    print('RUN #'+str(i))
    accs, margs = cross_validate(k=5, rejection_threshold=None, channel_runs=chan_runs)
    #print(accs, margs)
    accuracies.append(accs)
    margins.append(margs)

#print(accuracies, margins)
print('accuracies:', np.mean(accuracies, axis=0))
print('margins:', np.mean(margins, axis=0))
