import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import haversine_distances
from sklearn.base import BaseEstimator, TransformerMixin

# Create a function that we can re-use
# Source: https://docs.microsoft.com/es-mx/learn/modules/explore-analyze-data-with-python/5-exercise-visualize-data
def show_distribution(var_data, variable_name = "Data"):
    from matplotlib import pyplot as plt

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle(f'{variable_name} Distribution')
    
# KMeans Transformer
class KMeans_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, num_clusters: int, seed: int):
        self.num_clusters = num_clusters
        self.seed = seed
        self.cluster = KMeans(n_clusters = self.num_clusters, random_state = self.seed)
  
    def fit(self, X, y = None, **args):
        self.cluster.fit(X)
        return self

    def transform(self, X, y = None, **args):
        clusters = self.cluster.predict(X)
        return clusters[:, None]

    def get_feature_names(self):
        return ['neighbourhood']
    
# DBSCAN Transformer
# EXAMPLE
# >>kms_per_radian = 6371.0088 # Earth's radius
# >>db_transformer = DBSCAN_Transformer(epsilon = 1 / kms_per_radian, min_samples_dbscan = 5, algorithm_dbscan = 'ball_tree', \
# >>            metric_dbscan = 'haversine').fit(np.radians(X_train[coords_columns]))
class DBSCAN_Transformer(BaseEstimator, TransformerMixin):    
    def __init__(self, epsilon: int, min_samples_dbscan: int, algorithm_dbscan: int, metric_dbscan: int):
        self.epsilon = epsilon
        self.min_samples_dbscan = min_samples_dbscan
        self.algorithm_dbscan = algorithm_dbscan
        self.metric_dbscan = metric_dbscan
        self.cluster = DBSCAN(eps = epsilon, min_samples = min_samples_dbscan, 
                              algorithm = algorithm_dbscan, metric = metric_dbscan)

    def fit(self, X, y = None, **args):
        self.cluster.fit(X)
        return self

    def transform(self, X, y = None, **args):
        distances = haversine_distances(X, self.cluster.components_) # Distance between X_test and Core Points in X_train
        labels_train = self.cluster.labels_
        labels_test = []
        for dist in distances:
            index_min = np.argmin(dist)
            if dist[index_min] < self.epsilon:
                labels_test.append(labels_train[index_min])
            else:
                labels_test.append(-1)
        return np.array(labels_test)[:, None]

    def get_feature_names(self):
        return ['dbscan_clusters']
    