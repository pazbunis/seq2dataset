import numpy as np

class SeqSample:
    letter2num_lsb = {'A': 0, 'C': 1, 'G': 0, 'T': 1, 'N': 0}
    letter2num_msb = {'A': 0, 'C': 0, 'G': 1, 'T': 1, 'N': 0}
    num2letter = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    num2onehot = {0: [True, False, False, False],
                  1: [False, True, False, False],
                  2: [False, False, True, False],
                  3: [False, False, False, True]}

    def __init__(self, seq, header, label):
        idxs_lsb = list(map(lambda l: self.letter2num_lsb[l], seq))
        idxs_msb = list(map(lambda l: self.letter2num_msb[l], seq))

        self.two_bit = np.zeros((2, len(seq)), dtype=np.bool)
        self.two_bit[0, :] = idxs_lsb
        self.two_bit[1, :] = idxs_msb
        self.header = header
        self.label = label


    def seq(self):
        res = np.array([1, 2]).dot(self.two_bit)
        return ''.join([self.num2letter[n] for n in res])

    def onehot(self):
        res = np.array([1, 2]).dot(self.two_bit)
        out = np.eye(4, dtype=np.bool)[res]
        return out.transpose()
