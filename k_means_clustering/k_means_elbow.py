import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from utils import init_centroids, get_point_centroid_indices, compute_centroids, calculate_cost

if __name__ == "__main__":
    # Load data from the file
    data = sio.loadmat('k_means_clustering/data/dataset.mat')
    X = np.matrix(data['X'])

    # Initial setup
    K = 3
    max_iterations = 20
    max_clusters = 10
    centroids = init_centroids(X, K)

    # Elbow method (calculate optimal number of clusters)
    costs = np.zeros((max_clusters, 1))
    for c in range(1, max_clusters + 1):
        centroids = init_centroids(X, c)
        # Iterate through the centroids
        for i in range(0, max_iterations):
            indices = get_point_centroid_indices(X, centroids)
            centroids = compute_centroids(X, indices, c)
        costs[c - 1, 0] = calculate_cost(X, indices, centroids)

    # Plot the cost graph  
    plt.plot(np.arange(1, max_clusters + 1), costs, c='r', marker='o')
    plt.grid()
    plt.xlabel('Number of clusters')
    plt.ylabel('Cost')
    plt.show()


