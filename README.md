K-Means Clustering from Scratch
This repository contains a simple, self-contained Python implementation of the K-Means clustering algorithm. It demonstrates the core steps of K-Means, including centroid initialization, iterative assignment, and centroid updates, along with visualization of the clustering process.

## Features
Data Loading: Loads 2-dimensional data from a specified CSV/text file.

Centroid Initialization: Randomly initializes cluster centroids from the dataset.

Cluster Assignment: Assigns each data point to the nearest centroid.

Centroid Update: Recalculates centroids based on the mean of assigned data points.

Convergence Check: Iterates until cluster assignments no longer change.

Visualization: Plots the clusters and centroids at the initial, a middle, and the final converged iteration, saving the plots as PNG images.

Multiple K Values: Easily runnable for different numbers of clusters (k).

## Getting Started
Prerequisites
Before running the code, ensure you have the following Python libraries installed:

numpy

matplotlib

You can install them using pip:

pip install numpy matplotlib

Data File
The script expects a data file named ClusteringData.txt in the same directory as the Python script. This file should contain comma-separated numerical values, with at least two columns. The K-Means algorithm in this script will only use the first two columns for clustering.

Example ClusteringData.txt:

1.2,2.3,0.5
3.1,4.5,1.2
...

Running the Code
Save the provided Python code as kmeans_clustering.py (or any other .py file name).

Make sure your ClusteringData.txt file is in the same directory.

Run the script from your terminal:

python kmeans_clustering.py

## Usage
The KMeans function is called directly within the if __name__ == "__main__": block. By default, it will run the K-Means algorithm for k=2, k=3, and k=4 clusters.

You can modify the if __name__ == "__main__": block to test with different k values or a different dataset file:

if __name__ == "__main__":
    dataset_file = 'your_data.txt' # Change this if your file is different
    
    print("Running KMeans with K=5")
    KMeans(dataset_file, k=5)
    
    print("\nRunning KMeans with K=10")
    KMeans(dataset_file, k=10)

Output
Upon execution, the script will print messages indicating which K-Means run is in progress. For each k value, it will generate and save three PNG images:

kmeans_kX_iteration_1.png: Showing the initial cluster assignments.

kmeans_kX_iteration_Y.png: Showing the cluster assignments at a middle iteration (if applicable).

kmeans_kX_iteration_Z.png: Showing the final converged cluster assignments.

Where X is the value of k, Y is a middle iteration number, and Z is the final iteration number.

## Author
Anan Rahman 

## License
This project is open-source and available under the MIT License.
