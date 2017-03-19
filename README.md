# seq2dataset
Creates a dataset ready for running machine learning algorithms on it.

##Input:
    path_in: '.fastA' file with >headers and dna sequences
    path_out: target path
    target_length: a common length for all samples in the output. The sub-sequence will be taken from the middle.

##Output: 
    A dataset composed of SeqSample objects.
