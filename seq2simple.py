__author__ = 'pazbu'
"""
Input:
    path_in_positive: '.seq' file with "positive" dna sequences and their location
    path_in_negative: '.seq' file with "negative" dna sequences and their location
    path_out_X: target path for the samples
    path_out_y: target path for the labels
    target_length: a common length for all samples in the output. The sub-sequence will be taken from the middle.

Output:
    A dataset to be used for training or testing a machine learning model
"""
import sys
import math
import numpy as np
from structural_data import get_structural_params
import matplotlib.pyplot as plt

# Input params:
path_in_positive = 'CNNvsMOTIF/input/Enhancers.seq'
path_in_negative = 'CNNvsMOTIF/input/NEnhancers.seq'

path_out_train_X = 'CNNvsMOTIF/output/CNNvsMOTIF.train.X'
path_out_train_y = 'CNNvsMOTIF/output/CNNvsMOTIF.train.y'
path_out_validation_X = 'CNNvsMOTIF/output/CNNvsMOTIF.validation.X'
path_out_validation_y = 'CNNvsMOTIF/output/CNNvsMOTIF.validation.y'
path_out_test_X = 'CNNvsMOTIF/output/CNNvsMOTIF.test.X'
path_out_test_y = 'CNNvsMOTIF/output/CNNvsMOTIF.test.y'

target_length = 500
window_length = 1
train_ratio = 0.8
validation_ratio = 0
test_ratio = 1 - train_ratio - validation_ratio


def get_middle_subsequences(path_in):
    # collect sequences only (w/o the origin)
    lines = [line for line in open(path_in)]
    num_lines = len(lines)
    seq_lines = [lines[i].split('\t')[1] for i in range(0, num_lines)]

    # find shortest sequence length
    shortest_length = min([len(seq_lines[i]) for i in range(0, num_lines)])
    if shortest_length < target_length:
        sys.stderr.write('target sequence length is longer than the shortest sequence in the file.')
        exit(1)

    # extract the middle target_length characters (left-aligned in case of ties)
    seq_lines_mids = []
    for i in range(0, num_lines):
        l = len(seq_lines[i])
        start_idx = math.floor((l - target_length) // 2)
        for j in range(-window_length//2, window_length//2):
            seq_lines_mids.append(seq_lines[i][start_idx + j:start_idx + target_length +j])
    return seq_lines_mids


def dna_to_one_hot(seq):
    """converts a DNA sequence of length N to its one-hot 4xN representation"""
    seq = seq.upper()
    num2letter = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    letter2num = dict((v, k) for k, v in num2letter.items())
    num_bases = len(seq)
    letters = list(seq)
    idxs = list(map(lambda l: letter2num[l], letters))
    one_hot = np.zeros((4, num_bases))
    one_hot[idxs, np.arange(num_bases)] = 1
    return one_hot


def convert_samples_to_one_hot(raw_samples):
    samples = []
    for n in range(0, len(raw_samples)):
        # one_hot = dna_to_one_hot(raw_samples[n])
        # structural_params = get_structural_params(raw_samples[n])
        # try:
        #     concat = np.concatenate([one_hot, structural_params], axis=0)
        # except:
        #     print(n)
        #     print(raw_samples[n])
        #
        # samples.append(concat)
        samples.append(dna_to_one_hot(raw_samples[n]))
    return samples


def convert_labels_to_one_hot(raw_labels):
    labels = []
    label2one_hot = {0: (1, 0), 1: (0, 1)}
    for n in range(0, len(raw_labels)):
        labels.append(label2one_hot[raw_labels[n]])
    return labels


def reverse_sample(seqs):
    return [seq[::-1] for seq in seqs]

pos_Xs = get_middle_subsequences(path_in_positive)
neg_Xs = get_middle_subsequences(path_in_negative)
all_Xs = np.array(pos_Xs  + neg_Xs )
all_ys = np.array([1] * len(pos_Xs)  + [0] * len(neg_Xs))
perm = np.random.permutation(len(all_Xs))
all_Xs_shuffled = all_Xs[perm]
all_ys_shuffled = all_ys[perm]
samples = np.array(convert_samples_to_one_hot(all_Xs_shuffled))
labels = np.array(convert_labels_to_one_hot(all_ys_shuffled))

train_start_idx = 0
train_end_idx = math.ceil(len(all_Xs)*train_ratio)
validation_start_idx = train_end_idx
validation_end_idx = train_end_idx + math.ceil(len(all_Xs)*validation_ratio)
test_start_idx = validation_end_idx
test_end_idx = validation_end_idx + math.ceil(len(all_Xs)*test_ratio)

print(train_start_idx, train_end_idx, validation_start_idx, validation_end_idx, test_start_idx, test_end_idx)

np.save(path_out_train_X, samples[train_start_idx : train_end_idx])
np.save(path_out_train_y, labels[train_start_idx : train_end_idx])

np.save(path_out_validation_X, samples[validation_start_idx : validation_end_idx])
np.save(path_out_validation_y, labels[validation_start_idx : validation_end_idx])

np.save(path_out_test_X, samples[test_start_idx : test_end_idx])
np.save(path_out_test_y, labels[test_start_idx : test_end_idx])

test_pos_file = open('CNNvsMOTIF/output/test.pos', 'w')
test_neg_file = open('CNNvsMOTIF/output/test.neg', 'w')
for n in range(test_start_idx, test_end_idx-1):
    if all_ys_shuffled[n] == 1:
        test_pos_file.write("%s\n" % all_Xs_shuffled[n])
    else:
        test_neg_file.write("%s\n" % all_Xs_shuffled[n])
