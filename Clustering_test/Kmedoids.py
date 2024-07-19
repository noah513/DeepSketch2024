# K-medoids clustering using Hamming distance with frequencies (using actual data points as centers)

import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from collections import Counter
from scipy.spatial.distance import hamming

# Generate data
num_samples = 100  # Set small for testing purposes
num_features = 256
num_clusters = 10  # Set small for testing purposes
feature_length = 32

# Each data point has feature_length random features
data = []
for _ in range(num_samples):
    features = np.random.choice(num_features, feature_length, replace=True)
    feature_counts = Counter(features)
    data.append(feature_counts)

# Convert feature counts of each data point to a vector
data_matrix = np.zeros((num_samples, num_features), dtype=int)
for i, feature_counts in enumerate(data):
    for feature, count in feature_counts.items():
        data_matrix[i, feature] = count

# Define custom distance function
def custom_distance(x, y):
    count_distance = np.sum(np.abs(x - y))
    return count_distance

# Compute distance matrix
distance_matrix = pairwise_distances(data_matrix, metric=custom_distance)

# K-medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
kmedoids.fit(distance_matrix)

# Assign clusters
labels = kmedoids.labels_

medoid_indices = kmedoids.medoid_indices_
print(f"Medoid indices:\n{medoid_indices}")

cluster_centers = data_matrix[medoid_indices]
print(f"Cluster centers:\n{cluster_centers}")

print(f"Cluster centers (indices): {medoid_indices}")
print(f"Labels: {labels}")

# Data visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Dimensionality reduction to 2D using PCA
data_2d = PCA(n_components=2).fit_transform(data_matrix)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(data_2d[medoid_indices, 0], data_2d[medoid_indices, 1], c='red', s=200, marker='X')
plt.title('Clustering of Discrete Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()