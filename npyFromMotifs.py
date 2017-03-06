__author__ = 'pazbu'
import numpy as np

def dna_to_one_hot(seq):
    """converts a DNA sequence of length N to its one-hot 4xN representation"""
    seq = seq.upper()
    num2letter = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    letter2num = dict((v, k) for k, v in num2letter.items())
    num_bases = len(seq)
    letters = list(seq)
    idxs = list(map(lambda l: letter2num[l], letters))
    one_hot = np.zeros((4, num_bases), dtype=np.float32)
    one_hot[idxs, np.arange(num_bases)] = 1
    return one_hot

input_path = '/cs/grad/pazbu/paz/dev/projects/dnanet/data/CNNvsMOTIF/MOTIFS.N50.L5.txt'
output_path = '/cs/grad/pazbu/paz/dev/projects/dnanet/data/CNNvsMOTIF/MOTIFS.N50.L5'

with open(input_path,'r') as f:
    motifs = []
    for line in f:
        motifs.append(dna_to_one_hot(line.strip()))
    one_hot_size = motifs[0].shape

motifs_stack = np.reshape(np.vstack(motifs), one_hot_size + (1, len(motifs)))
np.save(output_path, motifs_stack)
