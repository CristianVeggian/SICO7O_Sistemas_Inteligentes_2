import numpy as np
import numpy.typing as npt

def normalize(X: list | npt.ArrayLike):
    if type(X) is list:
        X = np.array(X)
    
    media = X.mean()
    desvioPadrao = X.std()

    return np.array([(x-media)/desvioPadrao for x in X])