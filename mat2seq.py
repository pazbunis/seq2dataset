__author__ = 'pazbu'
import scipy.io
mat_path = '/cs/cbio/david/projects/CompGenetics/BaumWelch/peaks.mat'
path_out_pos_X = 'David.Enhancers.train.seq'

mat = scipy.io.loadmat(mat_path)
seqs = mat["seqs"][0]

with open(path_out_pos_X, 'w') as f:
    for line in seqs:
        f.write('>E_chrUnknown:unknown-unknown\t' + line[0].upper() + '\n')
