import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

from utils import split_data, accuracy

if __name__ == "__main__":
    # Load data from the file
    data = sio.loadmat('svm/data/ex6data1.mat')
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])

    # Initial setup
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    X_train, X_test, y_train, y_test = split_data(X, y) # stratified split
    alpha = 0.01

    theta = np.zeros((X.shape[1], 1))

    # Perform prediction
    for i in range(1, 1000):
        h = X_train.dot(theta)
        prod = np.multiply(h, y_train)
        count = 0
        for val in prod:
            if val >= 1:
                cost = 0
                theta = theta - alpha * (2 * 1/i * theta)
            else:
                cost = 1 - val 
                theta = theta + alpha * (np.multiply(X_train[count, :], y_train[count, :]).T - 2 * 1/i * theta)
            count += 1
    
    ## Prediction & accuracy
    y_pred = X_test.dot(theta)
    print('Test accuracy: %f' % (accuracy(y_test, y_pred)))

    # line = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    # h = - (np.multiply(theta[1, 0], line) + theta[0, 0]) / theta[2, 0]
    plt.scatter([X[:, 1]], [X[:, 2]], marker='o')
    # plt.plot(line, h, c='r')
    # plt.plot(X[:, 0], , c='r')
    plt.show()