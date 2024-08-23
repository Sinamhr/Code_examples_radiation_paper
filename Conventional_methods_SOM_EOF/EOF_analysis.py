import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
import os


# singular value decomposition (SVD) 
# input
# input data as anomaly relative to the whole dataset mean
# outputs
# U:left singular vectors as columns, S:The singular values, sorted in non-increasing order, Vt:ight singular vectors as rows
##U, S, Vt = svd(data_anomalies, full_matrices=False)



# U contains the temporal patterns (PC time series), shape: (time, modes)
# S contains the singular values, shape: (modes,)
# Vt contains the spatial patterns (EOFs), shape: (modes, space)

# we can calculate the explained variance by squaring the singular values and normalize them
# variance_explained = S**2 / np.sum(S**2)


# Extract the first 5 EOFs
# EOFs_5 = Vt[:num_eofs]  # Each row is an EOF pattern

# To get the PCs in a form where their variance matches the eigenvalues from a covariance matrix,
# multiply each column of U by the corresponding singular value
# Calculate the PCs for the first 5 modes
# PCs_5 = U[:, :num_eofs] * S[:num_eofs]






loaded = np.load('EOF_latent.npz')

PCs_5 = loaded['PCs_5']             # the PCs for the first 5 modes
EOFs_5 = loaded['EOFs_5']           # the first 5 EOFs




