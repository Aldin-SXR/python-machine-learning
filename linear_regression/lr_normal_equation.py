import scipy.io as sio
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Custom imports
from utils import *

def normal_equation(X, y):
    return inv(X.T.dot(X)) * (X.T.dot(y))

if __name__ == "__main__":
    # Load data from the file
    data = np.matrix(np.genfromtxt('linear_regression/data/data.txt', delimiter=','))
    X = data[:, 0:2]
    y = data[:, 2]
    X = normalize_features(X) # normalize features
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    # Calculate theta
    theta = normal_equation(X, y)
    print("Theta: [ %f, %f, %f ]" % (theta[0, 0], theta[1, 0], theta[2, 0]))

    # Calculate R^2 (statistical accuracy)
    r2 = r_squared(y, X.dot(theta.x.reshape(-1, 1)))
    print('R²: %f' % r2)