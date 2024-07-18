import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

# Step 1: Generate 256 random numbers (128 bytes)
random_bytes = np.random.randint(0, 256, 128, dtype=np.uint8)

# Step 2: Create 25,000 datasets with each dataset being a list of 32 random 128-byte values
num_datasets = 25000
num_values_per_dataset = 32
data_length = 128

datasets = []
for _ in range(num_datasets):
    dataset = [np.random.choice(random_bytes, data_length) for _ in range(num_values_per_dataset)]
    datasets.append(dataset)

# Step 3: Calculate pairwise distances based on the sum of Euclidean distances of corresponding indices
def custom_distance(a, b):
    dist = 0
    for i in range(len(a)):
        dist += euclidean(a[i], b[i])
    return dist

# Convert datasets to numpy array for efficiency
datasets_np = np.array(datasets)

# Perform K-means clustering with custom distance
kmeans = KMeans(n_clusters=100, random_state=0)

# Initialize centroids
initial_centroids = datasets_np[np.random.choice(datasets_np.shape[0], 100, replace=False)]

max_iterations = 100  # Set an arbitrary number of iterations
tolerance = 1e-5

for iteration in range(max_iterations):
    clusters = [[] for _ in range(100)]
    
    for dataset in datasets_np:
        distances = [custom_distance(dataset, centroid) for centroid in initial_centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(dataset)

    new_centroids = []
    for cluster in clusters:
        if cluster:
            new_centroid = np.mean(cluster, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(initial_centroids[len(new_centroids)])  # Keep the previous centroid if the cluster is empty

    new_centroids = np.array(new_centroids)
    
    # Check for convergence
    if np.allclose(initial_centroids, new_centroids, atol=tolerance):
        print(f"Iteration {iteration + 1}: Converged")
        break
    
    print(f"Iteration {iteration + 1}: Update centroids")
    initial_centroids = new_centroids

# Assign labels to each dataset based on the final centroids
kmeans.cluster_centers_ = initial_centroids
kmeans.labels_ = np.argmin([[custom_distance(dataset, centroid) for centroid in kmeans.cluster_centers_] for dataset in datasets_np], axis=1)

# Display the resulting cluster centers and labels
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels:\n", kmeans.labels_)