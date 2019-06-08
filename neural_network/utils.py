import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Normalize features
def normalize_features(X):
    return (X - np.mean(X, 0)) / np.std(X, 0)

# Calcualte R^2
def r_squared(y, yhat):
    sse = np.sum(np.power(y - yhat, 2))
    sst = np.sum(np.power(y - np.mean(y), 2))
    return 1 - sse / sst

# Calculate the sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

def confusion_matrix(y, yhat):
    p = (yhat > 0.5) * 1 # probability of class being 1
    conf = np.zeros((2, 2))
    for a, b in zip(y, p):
        if a == 1 and b == 1: # true positive
            conf[0, 0] += 1
        elif a == 1 and b == 0: # false negative
            conf[1, 0] += 1
        elif a == 0 and b == 1: # false positive
            conf[0, 1] += 1
        else: # true negative
            conf[1, 1] += 1
    return conf

# Plot the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, title=None):
    cm = cm.astype(int)
    cmap = plt.cm.get_cmap('Reds')
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Calculate prediction accuracy
def accuracy(theta, X, y):
    h = sigmoid(X.dot(theta))
    p = (h > 0.5) * 1
    return np.mean(p == y)

def multiclass_accuracy(theta, X, y):
    h = softmax(X.dot(theta))
    p = np.argmax(h, axis=1) + 1 # +1 due to indexing
    return np.mean(p == y)

def nn_accuracy(theta, X, y):
    m = X.shape[0]
    for i in range(0, theta.shape[0]):
        if i == 0:
            h = sigmoid(np.matrix(np.hstack([np.ones((m, 1)), X])) * theta[i, 0].T )
        else:
            h = sigmoid(np.matrix(np.hstack([np.ones((h.shape[0], 1)), h])) * theta[i, 0].T)
    p = np.argmax(h, axis=1) + 1 # +1 due to indexing
    return np.mean(p == y)

# Calculate classification precision
def precision(true_positives, predicted_positives):
    return true_positives / predicted_positives

# Calculate classification recall
def recall(true_positives, actual_positives):
    return true_positives / actual_positives

# Split data into train/test sets based on the percentage of target classes
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_test, y_train, y_test

# Softmax calculation for multiclass logistic regression
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1)

# 1-hot encoding of y-classes
def recode_y(y, classes):
    y_hot = np.zeros((y.shape[0], classes))
    for i in range(0, y_hot.shape[0]):
        y_hot[i, y[i] - 1] = 1
    return y_hot

# Randomly initialize the weight (theta) matrix
def rand_intialize_weights(l_in, num_h, h_out, l_out):
    epsilon_init = 0.12
    if num_h is not 0 and h_out > 0:
        w = np.random.rand(h_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
        w = w.reshape(-1, 1)
        for i in range(1, num_h + 1):
            if i == num_h:
                w_temp = np.random.rand(l_out, 1 + h_out) * 2 * epsilon_init - epsilon_init
                w = np.vstack([w, w_temp.reshape(-1, 1)])
            else:
                w_temp = np.random.rand(h_out, 1 + h_out) * 2 * epsilon_init - epsilon_init
                w = np.vstack([w, w_temp.reshape(-1, 1)])
    else:
        w = np.random.rand(l_out, 1 + l_in) * 2 * epsilon_init - epsilon_init
        w = w.reshape(-1, 1)
    # w[0:10000, :] = np.ones((10000, 1)) / 2
    # w[10000:, :] = np.ones((285, 1)) / 3
    return w

# Reshape theta into workable matrices
def reshape_theta(w, l_in, num_h, h_out, l_out):
    c = 0
    num_layers = num_h + 1
    theta = np.empty((num_layers, 1), dtype=object)
    for i in range(0, num_layers):
        if num_h is not 0 and h_out is not 0:
            if i == 0:
                length = np.arange(0, h_out * (l_in + 1))
                c = length[length.shape[0] - 1]
                theta[i, 0] = np.resize(w[length], (h_out, l_in + 1))
            elif i == num_layers - 1:
                if num_h == 1:
                    theta[i, 0] = np.resize(w[(h_out * (l_in + 1)):], (l_out, h_out + 1))
                else:
                    length = np.arange(c + 1, c + 1 + (h_out + 1) * l_out)
                    theta[i, 0] = np.resize(w[length], (l_out, h_out + 1))
            else:
                length = np.arange(c + 1, c + 1 + h_out * (h_out + 1))
                c = length[length.shape[0] - 1]
                theta[i, 0] = np.resize(w[length], (h_out, h_out + 1))
        else:
            theta[i, 0] = np.resize(w, (l_out, l_in + 1))
    return theta