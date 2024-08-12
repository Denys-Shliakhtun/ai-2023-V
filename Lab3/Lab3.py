import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.datasets import make_blobs

n_data = 500
cluster_centers = [[5, 1], [2, 7], [6, 4], [7, 9]]
blobs, _ = make_blobs(n_samples=n_data, centers=cluster_centers, cluster_std=1.0, random_state=69)
plt.scatter(blobs[:, 0], blobs[:, 1], s=20)
plt.show()

# Parameters for fuzzy cmeans
n_clusters = 4
m = 3 # Degree of fuzziness
centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(blobs.T, n_clusters, m, error=0.005, maxiter=50, init=None)
fuzzy_labels = np.argmax(u, axis=0)

# Plot clusters
for i in range(n_clusters):
    cluster_points = blobs[fuzzy_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', s=20)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, color='black')
plt.title('Clusters')
plt.legend()
plt.show()

#Centers of clusters
print("Centers of clusters:")
for center in centers:
    print(center)
plt.plot(jm)
plt.xlabel('Iterations')
plt.ylabel('Objective function values')
plt.title('Objective function values over iterations')
plt.grid(True)
plt.show()
