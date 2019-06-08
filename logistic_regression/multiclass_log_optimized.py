import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import normalize_features, confusion_matrix_c, plot_confusion_matrix, multiclass_accuracy, precision, recall, split_data, softmax, recode_y

# Calculate cost and gradient
def calculate_cost_and_gradient(theta, X, y, classes, l2_lambda):
    theta = theta.reshape(X.shape[1], classes)  # guarantees that theta will be a 2-d column array (necessary for minimize())                                    
    m = X.shape[0] # number of samples
    p = softmax(X.dot(theta))
    y_recoded = recode_y(y, classes) # 1-hot encoding of y
    cost = np.sum(np.multiply(y_recoded, -np.log(p))) / m + \
                l2_lambda / 2 * np.sum(np.power(theta[1:, :], 2))
    gradient = X.T.dot(p - y_recoded) / m  + l2_lambda * np.vstack([np.zeros((1, classes)), theta[1:, :] ])
    return cost, gradient

if __name__ == "__main__":
    # Load data from the file
    data = sio.loadmat('logistic_regression/data/digitsdata.mat')
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])
    # Initial setup
    # X = normalize_features(X) # normalize features
    classes = len(np.bincount(np.array(y).reshape(1, y.size)[0])) - 1
    X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = np.zeros((X.shape[1], classes))
    l2_lambda = 0

    # Calculate optimal theta using Newton Conjugate Gradient (closest alternative to fminunc)
    X_train, X_test, y_train, y_test = split_data(X, y) # stratified split
    theta = opt.minimize(fun=calculate_cost_and_gradient, x0=theta, method='TNC', args=(X_train, y_train, classes, l2_lambda), jac=True, options={'maxiter': 50})
    theta.x = theta.x.reshape(X.shape[1], classes)  # reshape theta into necessary dimensions                                

    # Print out the cost and calculated theta
    print('Testing cost: %f' % theta.fun)
    print('----------------------------------------------------') # separator
    
    # Calculate accuracy, precision and recall
    print('Accuracy (percentage of guessed samples): %f' % multiclass_accuracy(theta.x, X_test, y_test))
    conf = confusion_matrix_c(y_test, softmax(X_test.dot(theta.x)), True)
    plot_confusion_matrix(conf, 'Confusion matrix (actual vs predicted values)')

    print('Precision (true positives / predicted positives): %f' % precision(conf))
    print('Recall (true positives / actual positives): %f' % recall(conf))

    plt.show()
