"""
Determines the optimal number of clusters for k-means clustering in an unsupervised setting by quantifying the alignment
of clustering outcomes with an ideal scenario. This method operates on the principle that in an ideal clustering situation,
each set of data points consistently falls within the same cluster across multiple initializations of the k-means algorithm,
regardless of its random start.

The optimal cluster number is identified by selecting the count that minimizes this score (in our study 6 clusters), suggesting a stable and consistent
clustering performance. High-performing cluster numbers typically show significantly lower deviation scores, indicative of
more reliable and replicable clustering outcomes.
"""


import numpy as np
import os
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance



def WD_calculation(Frequency_matrix_row):
        """
        Calculate the Wasserstein Distance (WD) between a row of the frequency matrix and an ideal Dirac delta,
        which represents an ideal clustering scenario.
        #
        Parameters:
        frequency_matrix_row : numpy.ndarray
            A 1D array where each element records how often data points i and j are grouped together 
            in multiple k-means iterations.
        #
        Returns:
        float
            The Wasserstein Distance between the given distribution (row in the frequency matrix) and
            an ideal Dirac delta distribution representing perfect clustering.
        """
        n = Frequency_matrix_row.shape[0]
        x = np.array(range(1,n+1))
        actual_distribution = np.zeros((n))
        for i in range(n):    # Compute actual distribution based on frequency_matrix_row
            actual_distribution[i] = np.where(Frequency_matrix_row==x[i])[0].shape[0]
        ideal_distribution = np.zeros((n))  # Ideal Dirac delta distribution
        ideal_distribution[-1] = actual_distribution.sum()     # Ideal Dirac delta at the last position (1000 itteration)
        W_D = wasserstein_distance(x,x,ideal_distribution,actual_distribution)
        return W_D



def Optimal_clustering_number_measure(Latent_space_representation_C, Latent_space_representation_E, cluster_count, itteration=1000):
    """
    This function computes a clustering score for each potential cluster count using the following steps:
    1. Execute the k-means algorithm 1000 times on the data point representation (in our study on MCAE latent space which optimizes the
    recognition of large-scale patterns for our study purpose).
    2. Construct a frequency matrix for each cluster count, where each matrix element x(i,j) records how often data points i and j
    are grouped together in 1000 times k-mean clustering.
    3. Assess the ideal clustering scenario, where each matrix row would consist entirely of zeros and a single value of 1000,
    reflecting perfect consistency in clustering.
    4. Calculate the Wasserstein distance between the distribution of each row in the frequency matrix and an ideal Dirac delta
    function peaked at 1000. This distance measures how much the actual clustering deviates from the ideal scenario.
    5. Average these distances over all data points to derive a score for each cluster count, indicating the clustering's alignment 
    with the ideal.
    #
    Note: The function is not utilizing parallel processing for simplicity, but parallelization over the iterations is recommended for 
    performance optimization.
    #
    Parameters:
    Latent_space_representation_C : numpy.ndarray
        The data representation on which to perform clustering for the Control run data points.
    Latent_space_representation_E : numpy.ndarray
        The data representation on which to perform clustering for the Experiment run data points.
    cluster_count : int
        The number of clusters to evaluate.
    iterations : int, optional
        The number of times k-means is run, default is 1000.
    #
    Returns:
    float
        A score indicating the clustering's alignment with the ideal scenario for the specified cluster count.
    """
    Latent_space_representation  =  np.concatenate(( Latent_space_representation_C,  Latent_space_representation_E ), axis=0)
    Frequency_matrix = np.zeros((Latent_space_representation.shape[0], Latent_space_representation.shape[0]))
    for I in range(itteration):    # Perform k-means clustering multiple times and build the frequency matrix
        kmeans = KMeans(n_clusters=cluster_count) 
        kmeans.fit(Latent_space_representation)
        predict_label_C = kmeans.predict(Latent_space_representation_C)
        predict_label_E = kmeans.predict(Latent_space_representation_E)
        predict_label = np.concatenate(( predict_label_C,  predict_label_E ), axis=0)
        for i in range(cluster_count):         # Update the frequency matrix based on cluster assignments
            class_id = np.where(predict_label==i)[0]
            class_matrix = np.zeros((class_id.size, class_id.size))
            class_matrix += 1
            Frequency_matrix[class_id[:, None], class_id] += class_matrix
        del predict_label_C
        del predict_label_E
        del predict_label
    WD = np.zeros((Frequency_matrix.shape[0]))
    for i in range(Frequency_matrix.shape[0]):
        Frequency_matrix[i,i] = 0    
        WD[i] = WD_calculation(Frequency_matrix[i])
    cluster_score = WD.mean()
    return cluster_score
  
    




