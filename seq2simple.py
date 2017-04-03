import sys
import math
import numpy as np
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
path_in_positive = '/cs/grad/pazbu/paz/dev/projects/data/enhancers/cells/cerebellum.enhancers.fasta'
path_in_negative = '/cs/grad/pazbu/paz/dev/projects/seq2dataset/CNNvsMOTIF/input/NEnhancers.newline.seq'
path_out = '/cs/grad/pazbu/paz/dev/projects/data/enhancers/cerebellum/cerebellum.vs.not'
target_length = 500


def fastaread(fasta_name):
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


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
            sys.stderr.write('target sequence length is longer than a sequence in the file.\n')
            # exit(1)
        else:
            start_idx = math.floor((l - target_length) // 2)
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
            sys.stderr.write('target sequence length is longer than a sequence in the file.\n')
            # exit(1)
        else:
            for start_idx in range(0, l, target_length):
                if start_idx + target_length <= l:
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
    one_hot = np.zeros((4, num_bases), dtype=np.bool)
    one_hot[idxs, np.arange(num_bases)] = 1
    return one_hot


def reverse_sample(seqs):
    return [seq[::-1] for seq in seqs]

print('converting positives...')
samples = []
c = 0
for header, seq in middle_subseqs(path_in_positive):
    samples.append(SeqSample(seq, header, 'ENHANCER'))
    if c % 1000 == 0:
        print(c)
    c += 1

print('converting negatives...')
c = 0
for header, seq in all_subseqs(path_in_negative):
    samples.append(SeqSample(seq, header, 'BACKGROUND'))
    if c > 30000:
        break
    if c % 1000 == 0:
        print(c)
    c += 1

np.save(path_out, samples)
