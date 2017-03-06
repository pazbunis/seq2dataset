class SeqSample:
    def __init__(self, seq, onehot, header, label):
        self.seq = seq
        self.onehot = onehot
        self.header = header
        self.label = label