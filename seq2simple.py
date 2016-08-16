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
import numpy
# Input params:
path_in_positive = 'Enhancers.train.seq'
path_in_negative = 'NEnhancers.train.seq'
path_out_X = 'train_X.lines'
path_out_y = 'train_y.lines'
target_length = 500


def get_middle_subsequence(path_in):
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
        start_idx = math.floor((l - target_length) / 2)
        seq_lines_mids.append(seq_lines[i][start_idx:start_idx + target_length] + '\n')
    return seq_lines_mids

pos_Xs = get_middle_subsequence(path_in_positive)
neg_Xs = get_middle_subsequence(path_in_negative)
all_Xs = numpy.array(pos_Xs + neg_Xs)
all_ys = numpy.array([1] * len(pos_Xs) + [0] * len(neg_Xs))
perm = numpy.random.permutation(len(all_Xs))
all_Xs_shuffled = all_Xs[perm]
all_ys_shuffled = all_ys[perm]


# write outputs
with open(path_out_X, 'w') as f:
    for line in all_Xs_shuffled:
        f.write(line)

with open(path_out_y, 'w') as f:
    for line in all_ys_shuffled:
        f.write(str(line) + '\n')