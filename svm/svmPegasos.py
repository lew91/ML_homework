import numpy as np


def predict(w, x):
    return w * x.T

def batchPegasos(dataSet, labels, lam, T, k):
    m, n = np.shape(dataSet)
    w = np.zeros(n)
    dataaIndex = range(m)

    for t in range(1, T+1):
        wDelta = np.mat(np.zeros(n))
        eta = 1.0 / (lam * t)
        np.random.shuffle(dataaIndex)
        for j in range(k):
            i = dataaIndex[j]
            p = predict(w, dataSet[i, :])
            if labels[i] * p < 1:
                wDelta += labels[i] * dataSet[i, :].A

        w = (1.0 - 1 / t) * w + (eta / k)  * wDelta
    return w 
