__author__ = 'pazbu'
import numpy as np
import matplotlib.pyplot as plt
from math import floor

pentamers = [line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/sorted_5mers')]
ProT = np.array([line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/ProT')], dtype=np.float32)
MGW = np.array([line.rstrip('\n') for line in open('/cs/grad/pazbu/Desktop/combs/MGW')], dtype=np.float32)
HelT_a = np.array([line.rstrip('\n').split(',')[0] for line in open('/cs/grad/pazbu/Desktop/combs/HelT')], dtype=np.float32)
HelT_b = np.array([line.rstrip('\n').split(',')[1] for line in open('/cs/grad/pazbu/Desktop/combs/HelT')], dtype=np.float32)
Roll_a = np.array([line.rstrip('\n').split(',')[0] for line in open('/cs/grad/pazbu/Desktop/combs/Roll')], dtype=np.float32)
Roll_b = np.array([line.rstrip('\n').split(',')[1] for line in open('/cs/grad/pazbu/Desktop/combs/Roll')], dtype=np.float32)


ProT = (ProT - ProT.mean(axis=0)) / ProT.std(axis=0)
MGW = (MGW - MGW.mean(axis=0)) / MGW.std(axis=0)
HelT_a = (HelT_a - HelT_a.mean(axis=0)) / HelT_a.std(axis=0)
HelT_b = (HelT_b - HelT_b.mean(axis=0)) / HelT_b.std(axis=0)
Roll_a = (Roll_a - Roll_a.mean(axis=0)) / Roll_a.std(axis=0)
Roll_b = (Roll_b - Roll_b.mean(axis=0)) / Roll_b.std(axis=0)


pentamer_lu = {}
for n in range(0, len(ProT)):
    pentamer_lu[pentamers[n]] = (ProT[n], MGW[n], HelT_a[n], HelT_b[n], Roll_a[n], Roll_b[n])



def get_structural_params(seq):
    '''slides a 5bp window and generates a structure vector for the given sequence (uses 'A'-padding at start\end)'''
    window_size = 5
    pad_size = floor(window_size / 2.)
    lst = list(seq)

    padded_seq = ['A', 'A'] + lst + ['A', 'A']

    structure_vec = []
    for i in range(pad_size, len(padded_seq) - pad_size):
        pentamer = ''.join(padded_seq[i-pad_size:i+pad_size+1])
        structure_vec.append(pentamer_lu[pentamer])

    arr = np.array(structure_vec).transpose()
    return arr


# plt.plot(HelT_a)
# plt.show()