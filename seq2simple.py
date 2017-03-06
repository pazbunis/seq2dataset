__author__ = 'pazbu'
import sys
import math
import numpy as np
from seqsample import SeqSample
from itertools import groupby
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
path_in_positive = 'CNNvsMOTIF/input/Enhancers.newline.seq'
path_in_negative = 'CNNvsMOTIF/input/NEnhancers.newline.seq'
path_out = 'CNNvsMOTIF/output/dataset.8KPos.27KNeg'

target_length = 500

def fastaread(fasta_name):
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def middle_subseqs(path_in):
    faiter = fastaread(path_in)
    for header, seq in faiter:
        l = len(seq)
        if l < target_length:
            sys.stderr.write('target sequence length is longer than a sequence in the file.')
            exit(1)
        start_idx = math.floor((l - target_length) // 2)
        yield header, seq[start_idx:start_idx + target_length]


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


# def convert_samples_to_one_hot(raw_samples):
#     samples = []
#     for n in range(0, len(raw_samples)):
#         one_hot = dna_to_one_hot(raw_samples[n])
#         if np.shape(one_hot) != (4, target_length):
#             print(raw_samples[n])
#         samples.append(one_hot)
#     return samples


def convert_labels_to_one_hot(raw_labels):
    labels = []
    label2one_hot = {0: (1, 0), 1: (0, 1)}
    for n in range(0, len(raw_labels)):
        labels.append(label2one_hot[raw_labels[n]])
    return labels


def reverse_sample(seqs):
    return [seq[::-1] for seq in seqs]


samples = []
for header, seq in middle_subseqs(path_in_positive):
    onehot = dna_to_one_hot(seq)
    samples.append(SeqSample(seq, onehot, header, 'ENHANCER'))

for header, seq in middle_subseqs(path_in_negative):
    onehot = dna_to_one_hot(seq)
    samples.append(SeqSample(seq, onehot, header, 'BACKGROUND'))

np.save(path_out, samples)
