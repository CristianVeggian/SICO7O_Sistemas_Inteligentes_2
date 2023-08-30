import numpy as np
import numpy.typing as npt

def normalize(X: list | npt.ArrayLike):
    if type(X) is list:
        X = np.array(X)
    
    return np.linalg.norm(X)

