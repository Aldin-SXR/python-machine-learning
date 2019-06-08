import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from utils import init_centroids, get_point_centroid_indices, compute_centroids

if __name__ == "__main__":
    # Load data from the file
    data = sio.loadmat('k_means_clustering/data/dataset.mat')
    X = np.matrix(data['X'])

    # Initial setup
    K = 3
    max_iterations = 20
    centroids = init_centroids(X, K)
    colors = [ 'r', 'g', 'b', 'k', 'm' ]

    # Iterate through the centroids
    for i in range(0, max_iterations):
        plt.clf()
        indices = get_point_centroid_indices(X, centroids)
        centroids = compute_centroids(X, indices, K)
        # Plot by centroid iteration
        for k in range(0, K):
            idx = np.array(indices == k).flatten()
            plt.scatter([X[idx, 0]], [X[idx, 1]], c=colors[k], marker='o')
            plt.scatter([centroids[k, 0]], [centroids[k, 1]], c=colors[k], marker='x')
        plt.draw()
        plt.show(block=False)
        # Move on to next iteration (and exist once finished)
        if i is not max_iterations - 1:
            input("Press any key to continue...")
        else:
            plt.close()
    
    # Plot finalized clusters
    for i in range(0, K):
        idx = np.array(indices == i).flatten()
        plt.scatter([X[idx, 0]], [X[idx, 1]], c=colors[i], marker='o')
        plt.scatter([centroids[i, 0]], [centroids[i, 1]], c=colors[i], marker='x')

    plt.title('Finalized clusters')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()