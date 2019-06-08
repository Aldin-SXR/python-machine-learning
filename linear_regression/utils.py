# Normalize features
import numpy as np

def normalize_features(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)

def r_squared(y, yhat):
    sse = np.sum(np.power(y - yhat, 2))
    sst = np.sum(np.power(y - np.mean(y), 2))
    return 1 - sse / sst
