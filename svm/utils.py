from sklearn.model_selection import train_test_split
import numpy as np

# Split data into train/test sets based on the percentage of target classes
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    return X_train, X_test, y_train, y_test

def accuracy(y, yhat):
    p = np.zeros((y.shape[0], 1))         
    i = 0            
    for val in yhat:
        if val > 1:
            p[i, 0] = 1
        else:
            p[i, 0] = 0
        i += 1
    return np.mean(p.flatten() == np.array(y).flatten())
    