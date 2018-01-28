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
path_in_positive = '/cs/grad/pazbu/paz/dev/projects/data/ENCODE_hg19/combined.fa'
path_in_negative = '/cs/grad/pazbu/paz/dev/projects/data/ENCODE_mm10/dataset/negative_shuffled.fasta'
labels_path = '/cs/grad/pazbu/paz/dev/projects/data/ENCODE_hg19/combined.labels.tsv'
path_out = '/cs/grad/pazbu/paz/dev/projects/data/ENCODE_hg19/datasets/liver.thyroid'
target_length = 1000


def fastaread(fasta_name):
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def center_header(header, start_offset, length):
    chrom, location = header.split(':')
    added_info = ''
    if '|' in location:
        location, added_info = location.split('|', 1)
        added_info = '|' + added_info
    start, _ = location.split('-')
    start = int(start) + start_offset
    end = int(start) + length
    location = str(start) + '-' + str(end)
    return chrom + ':' + location + added_info


def middle_subseqs(path_in):
    """
    :param path_in:
    :return: the middle target_length letters from the sequences in the input file
    """
    faiter = fastaread(path_in)
    for header, seq in faiter:
        l = len(seq)
        seq = seq.upper()
        if l < target_length:
            pass
            #sys.stderr.write('target sequence length is longer than a sequence in the file.\n')
            # exit(1)
        else:
            start_idx = math.floor((l - target_length) // 2)
            header = center_header(header, start_idx, target_length)
            yield header, seq[start_idx:start_idx + target_length]


def all_subseqs(path_in):
    """
    :param path_in:
    :return: all disjoint segments of length target_length from the sequences in the input file
    """
    faiter = fastaread(path_in)
    for header, seq in faiter:
        l = len(seq)
        seq = seq.upper()
        if l < target_length:
            pass
            #sys.stderr.write('target sequence length is longer than a sequence in the file.\n')
            # exit(1)
        else:
            for start_idx in range(0, l, target_length):
                if start_idx + target_length <= l:
                    header = center_header(header, start_idx, target_length)
                    yield header, seq[start_idx:start_idx + target_length]


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


import pandas as pd
df = pd.read_csv(labels_path, sep='\t', header=None)
df = df[df.columns[3:]]
label_mat = df.as_matrix()
label_mat = label_mat.astype(int)


print('converting positives...')
samples = []
headers = []
labels = []
c = 0
d = dict()
for header, seq in middle_subseqs(path_in_positive):
    samples.append(dna_to_one_hot(seq))
    headers.append(header)
    labels.append(label_mat[c])

    rev_comp_seq = rev_comp(seq)
    samples.append(dna_to_one_hot(rev_comp_seq))
    headers.append(header + ' - revcomp')
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


print('number of positives (incl. rev-comps): ', len(samples))

# print('converting negatives...')
# c = 0

# neg_samples = []
# for header, seq in all_subseqs(path_in_negative):
# for header, seq in middle_subseqs(path_in_negative):
#     samples.append(dna_to_one_hot(seq))
#     headers.append(header)
#     labels.append(np.zeros(27))
#
#     rev_comp_seq = rev_comp(seq)
#     samples.append(dna_to_one_hot(rev_comp_seq))
#     headers.append(header + ' - revcomp')
#     labels.append(np.zeros(27))
#
#     aug_seq = augment(seq)
#     samples.append(dna_to_one_hot(aug_seq))
#     headers.append(header + ' - augmented')
#     labels.append(np.zeros(27))
#
#     rev_comp_aug_seq = rev_comp(aug_seq)
#     samples.append(dna_to_one_hot(rev_comp_aug_seq))
#     headers.append(header + ' - revcomp+augmented')
#     labels.append(np.zeros(27))
#     c += 1
#     if c % 1000 == 0:
#         print(c)
#
#     if c > 30000:
#         break

# print('number of negatives: ', len(neg_samples))
#
# ##################### MULTI-CLASS #########################
#
# # samples.extend(neg_samples)
samples_stacked = np.stack(samples)
labels = np.array(labels)
headers = np.array(headers)
###########################################################


# pos_labels = np.ones((len(samples), 1), dtype=bool)
# neg_labels = np.zeros((len(neg_samples), 1), dtype=bool)
# labels = np.vstack((pos_labels, neg_labels))
# samples.extend(neg_samples)
# samples_stacked = np.stack(samples)
# headers = np.array(headers)

# shuffle
idxs = np.arange(len(samples_stacked))
perm = np.random.permutation(idxs)
labels = labels[perm]
samples_stacked = samples_stacked[perm]
headers = headers[perm]

# divide
# train_index, validation_index, test_index = np.split(perm, [int(.8*len(perm)), int(0.85*len(perm))])

train_index = []
validation_index = []
test_index = []

for i, header in enumerate(headers):
    if 'chr6' in header or 'chr7' in header:
        validation_index.append(i)
    elif 'chr8' in header or 'chr9' in header:
        test_index.append(i)
    else:
        train_index.append(i)

# # compress
#
# # for (idxs, name) in zip((train_index, validation_index, test_index), ('train', 'validation', 'test')):
# #     Xr = np.reshape(samples_stacked[idxs], 4*1000*len(idxs))
# #     Xb = np.packbits(Xr)
# #     np.save('X_bin_' + name, Xb)
# #
# #     Yr = np.reshape(labels[idxs], 27*len(idxs))
# #     Yb = np.packbits(Yr)
# #     np.save('Y_bin_' + name, Yb)
# #
# #     np.save('headers_' + name, headers[idxs])
#
for (idxs, name) in zip((train_index, validation_index, test_index), ('train', 'validation', 'test')):
    np.save(os.path.join(path_out, 'X_' + name), samples_stacked[idxs])
    np.save(os.path.join(path_out, 'Y_' + name), labels[idxs])
    np.save(os.path.join(path_out, 'headers_' + name), headers[idxs])

print('after save')
#
# # np.save(path_out, samples)
# np.save(os.path.join(path_out, 'X_windows'), samples_stacked)
