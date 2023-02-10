import numpy as np

def SplitSeq(seq,n_steps):
    X,Y = [],[]
    for i in range(len(seq)):
        end_ix = i + n_steps
        if end_ix > len(seq)-1:
            break
        seq_x, seq_y = seq[i:end_ix], seq[end_ix]
        X.append(seq_x)
        Y.append(seq_y)
    return X,Y
