import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def calculate_delta_theta(theta, X, Y):
    return X.T.dot(X.dot(theta) - Y) / X.shape[0]

if __name__ == "__main__":
    # Load data from the file
    data = np.matrix(sio.loadmat("linear_regression/data/house_price.mat")["house_price"])
    X = data[:, 0]
    Y = data[:, 1]
    # Initial setup
    alpha = 0.0000001
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = np.zeros((X.shape[1], 1))
    # Calculate the changing theta
    while True:
        plt.clf()
        #Plot the original data set and hypothesis
        plt.scatter([X[:, 1]], [Y], label="Data points")
        h = X.dot(theta)
        plt.plot(X[:, 1], h, "r", label="Hypothesis")
        plt.legend(loc="upper left")
        plt.draw()
        plt.show(block=False)
        # Re-calculate theta
        theta = theta - alpha * calculate_delta_theta(theta, X, Y)
        print("Theta: [ %f, %f ]" % (theta[0, 0], theta[1, 0]))
        input("Press any key to continue...")