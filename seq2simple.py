__author__ = 'pazbu'
"""
Input:
    path_in: '.seq' file with dna sequences and their location
    path_out: target path
    target_length: a common length for all samples in the output. The sub-sequence will be taken from the middle.

Output: A dataset which has the same number of lines as the input file, but each line has raw dna of length target_length
"""
import sys
import math

# Input params:
path_in = 'Enhancers.test.seq'
path_out = 'Enhancers.test.lines'
target_length = 500

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

# write output
with open(path_out, 'w') as f:
    for line in seq_lines_mids:
        f.write(line)
