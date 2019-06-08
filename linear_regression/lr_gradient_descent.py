import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import normalize_features

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
    alpha = 0.1
    X = normalize_features(X) # normalize features
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = np.zeros((X.shape[1], 1))
    costs = np.array([])
    iterations = 0
    l2_lambda = 0.5
    # Calculate the changing theta
    while True:
        # Re-calculate theta
        cost, gradient = calculate_cost_and_gradient(theta, X, y, l2_lambda)
        theta = theta - alpha * gradient
        print('Cost: %f' % (cost))
        print("Theta: [ %f, %f, %f ]" % (theta[0, 0], theta[1, 0], theta[2, 0]))
        #Plot the dependence of cost vs iteration
        plt.clf()
        iterations += 1
        costs = np.hstack([costs, cost])
        plt.scatter(np.arange(1, iterations + 1), costs, label="Training cost", c='r')
        plt.legend(loc="upper right")
        plt.xticks(np.arange(0, 50, 5)) # pre-set ticks
        plt.draw()
        plt.show(block=False)
        # Move on to next iteration
        input("Press any key to continue...\n")