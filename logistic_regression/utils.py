import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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

def confusion_matrix_c(y, yhat, is_multi=False):
    if is_multi is True:
        p = np.argmax(yhat, axis=1) + 1 # +1 due to indexing
    else:
        p = (yhat > 0.5) * 1 # probability of class being 1
    conf = confusion_matrix(y, p)
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
    return np.mean(p == y.flatten())

def multiclass_accuracy(theta, X, y):
    h = softmax(X.dot(theta))
    p = np.argmax(h, axis=1) + 1 # +1 due to indexing
    return np.mean(p == y)

# Calculate classification precision
def precision(cm):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    return np.mean(precision)

# Calculate classification recall
def recall(cm):
    recall = np.diag(cm) / np.sum(cm, axis=1)
    return np.mean(recall)

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