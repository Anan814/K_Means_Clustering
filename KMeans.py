#Anan Rahman        1002035493

import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    
    data = np.loadtxt(file_name, delimiter=',', usecols=(0, 1))  # Only use first two columns
    return data

def initialize_centroids(data, k):
    
    random_indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[random_indices]
    return centroids

def assign_clusters(data, centroids):
    
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return centroids

def plot_clusters(data, labels, centroids, k, iteration):
    
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(data[labels == i][:, 0], data[labels == i][:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, color='black', linewidths=3, label='Centroids')
    plt.title(f'K = {k}, Iteration {iteration}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.savefig(f'kmeans_k{k}_iteration_{iteration}.png')  # Save the figure
    plt.show()

def KMeans(dataset_file, k=2):
    
    # Load data
    data = load_data(dataset_file)
    
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    
    # Variables to track iterations
    prev_labels = np.zeros(data.shape[0])
    iteration = 1
    total_iterations = 0
    plot_points = []  # Will store the iterations we want to plot
    
    while True:
        # Assign clusters
        labels = assign_clusters(data, centroids)
        
        # Store first iteration
        if iteration == 1:
            plot_points.append(iteration)
            plot_clusters(data, labels, centroids, k, iteration)
        
        # Update centroids
        new_centroids = update_centroids(data, labels, k)
        
        # Check for convergence
        if np.array_equal(labels, prev_labels):
            # Calculate middle iteration
            middle_iteration = (iteration + 1) // 2
            if middle_iteration not in plot_points and middle_iteration != iteration:
                plot_points.append(middle_iteration)
                # Need to recreate the state at middle iteration
                temp_centroids = centroids.copy()
                temp_labels = labels.copy()
                # Run iterations up to middle point
                for i in range(1, middle_iteration):
                    temp_labels = assign_clusters(data, temp_centroids)
                    temp_centroids = update_centroids(data, temp_labels, k)
                plot_clusters(data, temp_labels, temp_centroids, k, middle_iteration)
            
            # Store final iteration
            plot_points.append(iteration)
            plot_clusters(data, labels, centroids, k, iteration)
            break
        
        prev_labels = labels
        centroids = new_centroids
        iteration += 1
        total_iterations += 1

# Running the KMeans algorithm with different k values
if __name__ == "__main__":
    dataset_file = 'ClusteringData.txt'
    
    # Run for default K = 2
    print("Running KMeans with K=2")
    KMeans(dataset_file, k=2)
    
    # Run for K = 3
    print("\nRunning KMeans with K=3")
    KMeans(dataset_file, k=3)
    
    # Run for K = 4
    print("\nRunning KMeans with K=4")
    KMeans(dataset_file, k=4)