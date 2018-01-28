import math
import os
import numpy as np
from dnautils import rev_comp
import random
import dnautils
from seqsample import SeqSample
from itertools import groupby
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

# Input params:
path_in_seqs = '/cs/grad/pazbu/paz/dev/projects/dna-simulator/datasets/single_motif/seqs.npy'
path_in_labels = '/cs/grad/pazbu/paz/dev/projects/dna-simulator/datasets/single_motif/labels.npy'
path_out = '/cs/grad/pazbu/paz/dev/projects/dna-simulator/datasets/single_motif/ds'

seqs = np.load(path_in_seqs)
label_mat = np.load(path_in_labels)


def dna_to_one_hot(seq):
    """converts a DNA sequence of length N to its one-hot 4xN representation"""
    seq = seq.upper()
    num2letter = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    letter2num = dict((v, k) for k, v in num2letter.items())
    letter2num['N'] = 0
    num_bases = len(seq)
    letters = list(seq)
    idxs = list(map(lambda l: letter2num[l], letters))
    one_hot = np.zeros((4, num_bases), dtype=np.uint8)
    one_hot[idxs, np.arange(num_bases)] = 1
    return one_hot


def augment(seq):
    shift = 0
    pad_char = 'A'
    while shift == 0:
        shift = random.randint(-50, 50)
    if shift < 0:
        aug_seq = abs(shift) * pad_char + seq[:len(seq) - abs(shift)]
    else:
        aug_seq = seq[shift:] + abs(shift) * pad_char
    return aug_seq


print('converting positives...')
samples = []
headers = []
labels = []
c = 0
d = dict()
for seq in seqs:
    samples.append(dna_to_one_hot(seq))
    headers.append('simulation')
    labels.append(label_mat[c])

    rev_comp_seq = rev_comp(seq)
    samples.append(dna_to_one_hot(rev_comp_seq))
    headers.append('simulation - revcomp')
    labels.append(label_mat[c])

    # aug_seq = augment(seq)
    # samples.append(dna_to_one_hot(aug_seq))
    # headers.append(header + ' - augmented')
    # labels.append(label_mat[c])
    #
    # rev_comp_aug_seq = rev_comp(aug_seq)
    # samples.append(dna_to_one_hot(rev_comp_aug_seq))
    # headers.append(header + ' - revcomp+augmented')
    # labels.append(label_mat[c])

    if c % 1000 == 0:
        print(c)
    c += 1

samples_stacked = np.stack(samples)
labels = np.array(labels)
headers = np.array(headers)


# shuffle
idxs = np.arange(len(samples_stacked))
perm = np.random.permutation(idxs)
labels = labels[perm]
samples_stacked = samples_stacked[perm]
headers = headers[perm]

# divide
train_index, validation_index, test_index = np.split(perm, [int(.8*len(perm)), int(0.85*len(perm))])

for (idxs, name) in zip((train_index, validation_index, test_index), ('train', 'validation', 'test')):
    np.save(os.path.join(path_out, 'X_' + name), samples_stacked[idxs])
    np.save(os.path.join(path_out, 'Y_' + name), labels[idxs])
    np.save(os.path.join(path_out, 'headers_' + name), headers[idxs])

print('after save')
#
# # np.save(path_out, samples)
# np.save(os.path.join(path_out, 'X_windows'), samples_stacked)
