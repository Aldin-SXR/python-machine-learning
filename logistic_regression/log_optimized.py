import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import normalize_features, sigmoid, confusion_matrix_c, plot_confusion_matrix, accuracy, precision, recall, split_data

# Calculate cost and gradient
def calculate_cost_and_gradient(theta, X, y, l2_lambda):
    theta = theta.reshape(-1, 1)  # guarantees that theta will be a 2-d column array (necessary for minimize())                                    
    m = X.shape[0] # number of samples
    cost = np.sum(np.multiply(y, np.log(sigmoid(X.dot(theta)))) + np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta))))) / -m + \
                l2_lambda / (2 * m) * np.sum(np.power(theta[1:], 2)) / (2 * m) # cost + regularization parameter
    gradient = X.T.dot(sigmoid(X.dot(theta)) - y) / m + l2_lambda / m * np.vstack([0, theta[1:]])
    return cost, gradient

if __name__ == "__main__":
    # Load data from the file
    data = np.matrix(np.genfromtxt('logistic_regression/data/data.csv', delimiter=','))
    X = data[:, 0:2]
    y = data[:, 2]
    # Initial setup
    X = normalize_features(X) # normalize features
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = np.zeros((X.shape[1], 1))
    l2_lambda = 0

    # Calculate optimal theta using Newton Conjugate Gradient (closest alternative to fminunc)
    X_train, X_test, y_train, y_test = split_data(X, y) # stratified split
    theta = opt.minimize(fun=calculate_cost_and_gradient, x0=theta, method='TNC', args=(X_train, y_train, l2_lambda), jac=True)

    # Print out the cost and calculated theta
    print('Cost: %f' % theta.fun)
    print('Theta: [ %f, %f, %f ]' % (theta.x[0], theta.x[1], theta.x[2]))
    print('----------------------------------------------------') # separator
    
    # Calculate accuracy, precision and recall
    print('Accuracy (percentage of guessed samples): %f' % accuracy(theta.x, X_test, y_test))
    conf = confusion_matrix_c(y_test, sigmoid(X_test.dot(theta.x)).T)
    plot_confusion_matrix(conf, 'Confusion matrix (actual vs predicted values)')

    print('Precision (true positives / predicted positives): %f' % precision(conf))
    print('Recall (true positives / actual positives): %f' % recall(conf))

    plt.show()
