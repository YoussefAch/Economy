import numpy as np
import pandas as pd

def generatePossibleSequences(nbIntervals, order):

    """
       This function return the possible sequences of intervals given order
       and number of intervals

    """
    seqBase = np.arange(nbIntervals)
    final = []
    for i in range(order):
        if i==0:
            # repeat
            final.append(np.array([seqBase for _ in range(nbIntervals**(order-1))]).flatten())
        else:
            #duplicate
            seq = np.array([(nbIntervals**i)*[e] for e in seqBase]).flatten()
            #repeat
            final.append(np.array([seq for _ in range(nbIntervals**(order-i-1))]).flatten())
    return np.array(final)

def transform_to_format_fears(X):
    nbObs, length = X.shape
    for i in range(nbObs):
        ts = X.iloc[i,:]
        data = {'id':[i for _ in range(length)], 'timestamp':[k for k in range(1,length+1)], 'dim_X':list(ts.values)}
        if i==0:
            df = pd.DataFrame(data)
        else:
            df = df.append(pd.DataFrame(data))
    df = df.reset_index(drop=True)
    return df

def numpy_to_df(x):
    return pd.DataFrame({i+1:e for i,e in enumerate(x)}, index=range(1))
