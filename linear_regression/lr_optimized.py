# Optimized version of linear regression (using scipy)
import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import *

# Calculate cost and gradient
def calculate_cost_and_gradient(theta, X, y, l2_lambda):
    theta = theta.reshape(-1, 1)  # guarantees that theta will be a 2-d column array (necessary for minimize())                                    
    m = X.shape[0] # number of samples
    cost = (np.sum(np.power(X.dot(theta) - y, 2)) + \
                l2_lambda * np.sum(np.power(theta[1:], 2))) / (2 * m) # cost + regularization parameter
    gradient = X.T.dot(X.dot(theta) - y) / m + l2_lambda / m * np.vstack([0, theta[1:]])
    return cost, gradient

if __name__ == "__main__":
    # Load data from the file
    data = np.matrix(np.genfromtxt('linear_regression/data/data.txt', delimiter=','))
    X = data[:, 0:2]
    y = data[:, 2]
    # Initial setup
    X = normalize_features(X) # normalize features
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = np.zeros((X.shape[1], 1))
    l2_lambda = 0.5 # Lambda regularization parameter
    # Calculate optimal theta using Newton Conjugate Gradient (closest alternative to fminunc)
    theta = opt.minimize(fun=calculate_cost_and_gradient, x0=theta, method='TNC', args=(X, y, l2_lambda), jac=True)

    # Print out the cost and calculated theta
    print('Cost: %f' % theta.fun)
    print("Theta: [ %f, %f, %f ]" % (theta.x[0], theta.x[1], theta.x[2]))

    # Calculate R^2 (statistical accuracy)
    r2 = r_squared(y, X.dot(theta.x.reshape(-1, 1)))
    print('R²: %f' % r2)