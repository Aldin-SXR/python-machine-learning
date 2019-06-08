import numpy as np

# Initialize centroid points
def init_centroids(X, K):
    centroids = np.zeros((K, X.shape[1]))
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[0:K], :]
    return centroids

# Return the cluster indices
def get_point_centroid_indices(X, centroids):
    K = centroids.shape[0]
    indices = np.zeros((X.shape[0], 1))
    m = X.shape[0]

    # Calculate distances to each point
    distances = np.zeros((K, 1))
    for i in range(0, m):
        for j in range(0, K):
            distances[j, 0] = np.sum(np.power(centroids[j, :] - X[i, :], 2))
        indices[i, 0] = np.argmin(distances)

    return indices

# Computer new centroids
def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))

    # Calculate new centroids
    for i in range(0, K):
        temp_x = X[np.array(idx == i).flatten(), :]
        centroids[i, :] = np.mean(temp_x, axis=0)

    return centroids 

# Calculate K-means cost
def calculate_cost(X, idx, centroids):
    K = centroids.shape[0]
    m = X.shape[0]
    cost = 0
    
    for i in range(0, m):
        index = np.array(idx[i], dtype=int)
        cost = cost + np.sum(np.power(X[i, :] - centroids[index, :], 2))
    
    cost = np.sum(cost) / m
    return cost