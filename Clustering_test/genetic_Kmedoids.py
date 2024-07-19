# Genetic Algorithm to Minimize Distance by Maximizing Common Features

import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import pairwise_distances
from collections import Counter
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from deap import base, creator, tools, algorithms

# Data generation
num_samples = 100  # Set small for testing purposes, can increase if needed
num_features = 256
num_clusters = 10  # Set small for testing purposes, can increase if needed
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

# Calculate Hamming distances
def calculate_hamming_distances(data_matrix):
    num_samples = data_matrix.shape[0]
    hamming_distances = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            hamming_distances[i, j] = hamming(data_matrix[i], data_matrix[j])
            hamming_distances[j, i] = hamming_distances[i, j]
    return hamming_distances

hamming_distances = calculate_hamming_distances(data_matrix)

# Calculate common feature counts
def calculate_common_features(data_matrix):
    num_samples = data_matrix.shape[0]
    common_feature_counts = np.zeros((num_samples, num_samples), dtype=int)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            common_features = np.minimum(data_matrix[i], data_matrix[j])
            common_feature_counts[i, j] = np.sum(common_features)
            common_feature_counts[j, i] = common_feature_counts[i, j]
    return common_feature_counts

common_feature_counts = calculate_common_features(data_matrix)

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    adjustment_factor = individual[0]
    adjusted_distances = np.zeros_like(hamming_distances)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            adjusted_distance = hamming_distances[i, j] / (1 + adjustment_factor * common_feature_counts[i, j])
            adjusted_distances[i, j] = max(0, adjusted_distance)  # Ensure non-negative distances
            adjusted_distances[j, i] = adjusted_distances[i, j]
    
    kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
    kmedoids.fit(adjusted_distances)
    
    labels = kmedoids.labels_
    
    # Clustering performance evaluation metric (e.g., silhouette score, number of clusters, etc.)
    unique_labels = np.unique(labels)
    if len(unique_labels) < num_clusters:
        return (float('inf'),)  # Penalty for insufficient number of clusters
    
    # Simple example of evaluating whether all clusters have similar sizes
    cluster_sizes = np.array([np.sum(labels == label) for label in unique_labels])
    size_std = np.std(cluster_sizes)
    
    return (size_std,)  # Minimize the standard deviation of cluster sizes

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=True)

# Select the best individual
best_individual = tools.selBest(population, k=1)[0]
optimal_adjustment_factor = best_individual[0]
print(f"Optimal adjustment factor: {optimal_adjustment_factor}")

# Generate the optimal adjusted distance matrix
adjusted_distances = np.zeros_like(hamming_distances)
for i in range(num_samples):
    for j in range(i + 1, num_samples):
        adjusted_distances[i, j] = hamming_distances[i, j] / (1 + optimal_adjustment_factor * common_feature_counts[i, j])
        adjusted_distances[j, i] = adjusted_distances[i, j]

# K-medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, metric='precomputed', random_state=42)
kmedoids.fit(adjusted_distances)

# Assign clusters
labels = kmedoids.labels_

# Medoid indices
medoid_indices = kmedoids.medoid_indices_
print(f"Medoid indices:\n{medoid_indices}")

# Cluster centers
cluster_centers = data_matrix[medoid_indices]
print(f"Cluster centers:\n{cluster_centers}")

# Cluster centers and assigned labels
print(f"Cluster centers (indices): {medoid_indices}")
print(f"Labels: {labels}")

# Data visualization

# Dimensionality reduction to 2D using PCA
data_2d = PCA(n_components=2).fit_transform(data_matrix)

plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(data_2d[medoid_indices, 0], data_2d[medoid_indices, 1], c='red', s=200, marker='X')
plt.title('Clustering of Discrete Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
