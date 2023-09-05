import numpy as np
import numpy.typing as npt

def normalize(X: list | npt.ArrayLike):
    if type(X) is list:
        X = np.array(X)
    
    media = X.mean()
    desvioPadrao = X.std()

    return np.array([(x-media)/desvioPadrao for x in X])

def confusionMatrix(y_real: list | npt.ArrayLike, y_predict: list | npt.ArrayLike):

    output = {"tp":0, "fp":0, "tn":0, "fn":0}

    print(y_real)
    print(y_predict)

    for yr, yp in zip(y_real, y_predict):
        if yr == yp and yr == 1:
            output["tp"] += 1
        elif yr == yp and yr == -1:
            output["tn"] += 1
        elif yr != yp and yr == -1:
            output["fp"] += 1
        else:
            output["fn"] += 1
    return output