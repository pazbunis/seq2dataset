# seq2dataset
Creates a dataset ready for machine learning based on a '.seq' file.

##Input:
    path_in: '.seq' file with dna sequences and their location
    path_out: target path
    target_length: a common length for all samples in the output. The sub-sequence will be taken from the middle.

##Output: 
    A dataset which has the same number of lines as the input file, but each line has raw dna of length target_length
