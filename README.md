

2. Clustering Algorithm Implementation 
 KMeans Clustering 
KMeans clustering is an iterative, partition-based clustering algorithm that splits the data into a predefined number of clusters. The steps involved are:

Initializing k centroids randomly.
Assigning each data point to the nearest centroid, forming clusters.
Updating each centroid based on the mean of the points in its cluster.
Repeating the process until centroids no longer change.
Suitability of KMeans for the Iris Dataset
The Iris dataset is compact and well-separated, with 3 known species, making KMeans clustering suitable. The low-dimensionality (4 features) helps KMeans to work effectively, as it often performs well on smaller datasets with distinct clusters.

Applying KMeans Clustering and Visualizing the Clusters

python
Copy code
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Apply KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(df)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['kmeans_cluster'], palette='viridis', s=100)
plt.title("KMeans Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(title='Cluster')
plt.show()
 B: Hierarchical Clustering 

Hierarchical clustering is a method that builds a hierarchy of clusters, typically represented as a dendrogram. It comes in two forms:

Agglomerative: Starts with each data point as its own cluster and iteratively merges clusters.
Divisive: Starts with all points in a single cluster and iteratively splits them.
The agglomerative approach with a dendrogram can be used to choose the number of clusters by observing where to cut the dendrogram.

Suitability of Hierarchical Clustering for the Iris Dataset
Hierarchical clustering is ideal for small datasets like Iris, as it provides an intuitive visualization of clusters through a dendrogram. This approach is useful when the number of clusters isn’t known beforehand, as the dendrogram can help identify the natural cluster separation.

Applying Hierarchical Clustering and Visualizing the Clusters

python
Copy code
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Apply Agglomerative Clustering with 3 clusters
hierarchical = AgglomerativeClustering(n_clusters=3)
df['hierarchical_cluster'] = hierarchical.fit_predict(df)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], hue=df['hierarchical_cluster'], palette='coolwarm', s=100)
plt.title("Hierarchical Clustering on Iris Dataset")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend(title='Cluster')
plt.show()

# Dendrogram for further analysis
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(df.iloc[:, :4], method='ward')
dendrogram(linkage_matrix)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()








