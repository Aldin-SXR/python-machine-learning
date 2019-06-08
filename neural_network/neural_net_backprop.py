import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from utils import rand_intialize_weights, sigmoid, sigmoid_gradient, confusion_matrix, plot_confusion_matrix, nn_accuracy, precision, recall, split_data, recode_y, reshape_theta

# Calculate cost and gradient
def calculate_cost_and_gradient(theta, X, y, l2_lambda, hidden_layers, input_layer_size, hidden_layer_size, classes):
    theta = reshape_theta(theta, input_layer_size, hidden_layers, hidden_layer_size, classes)
    # Initial setup
    num_layers = hidden_layers + 1
    m = X.shape[0] # number of samples
    cost = 0
    theta_grad = np.empty((num_layers, 1), dtype=object)

    # Recode y output
    y = recode_y(y, classes)

    # Feed-forward the neural network
    for i in range(0, m):
        for l in range(0, num_layers):
            if l == 0: # if dealing with the first layer, use X values
                h = sigmoid(np.matrix(np.append(1, X[i, :])).dot(theta[l, 0].T))
                # h = np.matrix(np.hstack([1, X[i, :]])).dot(theta[l, 0].T)
            else:
                h = sigmoid(np.matrix(np.append(1, h)).dot(theta[l, 0].T))
        for k in range(0, classes):
            cost = cost + (-y[i, k] * np.log(h[0, k]) - (1 - y[i, k]) * np.log(1 - h[0, k]))
    cost = cost / m

    # Back-propagation algorithm
    delta = np.empty((num_layers, 1), dtype=object)
    for i in range(0, num_layers):
        delta[i, 0] = 0

    for t in range(0, m):
        # Activations of all layers
        a = np.empty((num_layers + 1, 1), dtype=object)
        z = np.empty((num_layers, 1), dtype=object)
        for i in range(0, num_layers + 1):
            if i == 0:
                a[i, 0] = np.matrix(np.append(1, X[t, :]))
            else:
                z[i - 1, 0] = np.matrix(a[i - 1, 0]).dot(theta[i - 1, 0].T)
                if i == num_layers:
                    a[i, 0] = sigmoid(z[i - 1, 0])
                else:
                    a[i, 0] = np.matrix(np.append(1, sigmoid(z[i - 1, 0])))
        
        # Layer deltas
        d = np.empty((num_layers, 1), dtype=object)
        for i in range(num_layers - 1, -1, -1):
            if i == num_layers - 1:
                d[i, 0] = np.zeros((classes, 1))
                for k in range(0, classes):
                    d[i, 0][k, 0] = a[num_layers, 0][0, k] - y[t, k]
            else:
                d[i, 0] = np.multiply(theta[i + 1, 0][:, 1:].T.dot(d[i + 1, 0]), sigmoid_gradient(z[i, 0].T))

        for i in range(0, num_layers):
            delta[i, 0] = delta[i, 0] + d[i, 0].dot(a[i, 0])

    for i in range(0, num_layers):
        delta[i, 0] = delta[i, 0] / m

    # Cost regularization
    reg = np.zeros((num_layers, 1))
    for i in range(0, num_layers):
        reg[i, 0] = np.sum(np.power(theta[i, 0], 2))
    
    cost = cost + l2_lambda / (2 * m) * np.sum(reg)

    # Gradient + gradient regularization
    for i in range(0, num_layers):
        theta_grad[i, 0] = np.matrix(np.hstack([delta[i, 0][:, 0], delta[i, 0][:, 1:] + (l2_lambda / m) * theta[i, 0][:, 1:]]))

    # Unroll gradient
    gradient = np.empty((0, 1))
    for i in range(0, num_layers):
        gradient = np.vstack([ gradient, theta_grad[i, 0].ravel().T ])
    
    return cost, gradient

if __name__ == "__main__":
    # Load data from the file
    data = sio.loadmat('neural_network/data/digitsdata.mat')
    X = np.matrix(data['X'])
    y = np.matrix(data['y'])
    # Initial setup
    input_layer_size = X.shape[1] # number of features
    hidden_layer_size = 15
    hidden_layers = 2 # number of hidden layers
    num_layers = hidden_layers + 1 # number of total layers (input + hidden)
    classes = len(np.bincount(np.array(y).reshape(1, y.size)[0])) - 1

    # X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    theta = rand_intialize_weights(input_layer_size, hidden_layers, hidden_layer_size, classes)
    l2_lambda = 0.1

    # Calculate optimal theta using Newton Conjugate Gradient (closest alternative to fminunc)
    X_train, X_test, y_train, y_test = split_data(X, y) # stratified split
    theta = opt.minimize(fun=calculate_cost_and_gradient, x0=theta, \
                                          method='TNC', args=(X, y, l2_lambda, hidden_layers, input_layer_size, hidden_layer_size, classes), \
                                          jac=True, options={'maxiter': 10, 'disp': True})
    
    theta.x = reshape_theta(theta.x, input_layer_size, hidden_layers, hidden_layer_size, classes)                    

    # Print out the cost and calculated theta
    print('\nTesting cost: %f' % theta.fun)
    for i in range(0, theta.x.shape[0]):
        print('Theta%d dimensions: [ %d x %d ]' % (i, theta.x[i, 0].shape[0], theta.x[i, 0].shape[1]))
    # print('Accuracy (percentage of guessed samples): %f' % nn_accuracy(theta.x, X, y))
    print('----------------------------------------------------') # separator
    
    # Calculate accuracy, precision and recall
    # conf = confusion_matrix(y_test, sigmoid(X_test.dot(theta.x)).T)
    # plot_confusion_matrix(conf, 'Confusion matrix (actual vs predicted values)')

    # print('Precision (true positives / predicted positives): %f' % precision(conf[0, 0], conf[0, 0] + conf[0, 1]))
    # print('Recall (true positives / actual positives): %f' % recall(conf[0, 0], conf[0, 0] + conf[1, 0]))

    plt.show()
