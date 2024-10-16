# Arctic Climate Response to European Radiative Forcing: A Deep Learning Study on Circulation Pattern Changes

## Overview

This repository contains the example code and input files used in our paper:

**Title:** Arctic Climate Response to European Radiative Forcing: A Deep Learning Study on Circulation Pattern Changes  
**Authors:** Sina Mehrdad et al.



## Repository Contents

### `Auto_Encoders` Directory
A directory consist of the trained Autoencoders used in this study with their example input files. the directory include a code `DL_Model_test.py` that show an example used of the autoencoders
#### `DL_Model_test.py`
- **Description:** Loads the trained deep learning models presented in the paper.
- **Functionality:** Demonstrates how to use the model with example data points provided in the repository.
- **Note:** the autoencoders can be used for other analysis puposes.

### `Class_Contribution_calculation` Directory
A directory consist an example code of how to calculate class contribution to the anomaly, the code is implemented on the Sea Ice Concentration (SIC) field as an example. and the directory include the daily field of SIC for the Control and Experiment run in npz numpy saving format
#### `SIC_class_contribution.py`
- **Description:** Calculates the class contributions for Sea Ice Concentration (SIC) as described in the paper.
- **Note:** The procedure can be easily adapted for other fields using similar methods.

### `Conventional_methods_SOM_EOF/` Directory
Contains code for the implementation of conventional methods, including Self-Organizing Maps (SOM) and Empirical Orthogonal Function (EOF) analysis.
#### `SOM_tf.py`
- **Description:** SOM implementation using Tensorflow
#### `EOF_analysis`
- **Description:** EOF analysis code example

### `Optimal_cluster_measure` Directory
This directory contains code that we developed to determine the optimal number of clusters based on the distribution of data points in the latent representation.
#### `Optimal_clustering_number_measure.py`
- **Description:** This script determines the optimal number of clusters in an unsupervised setting by quantifying how well the clustering outcomes align with an ideal scenario. The method operates on the principle that, in an ideal clustering situation, each set of data points consistently falls within the same cluster across multiple initializations of the k-means algorithm, regardless of its random start.
- **Note:** This code computes a clustering score for each potential cluster count. The optimal cluster number is identified by calculating the score for different cluster counts and the cluster count with the minimum scorce is the optimal cluster count. this minimum suggesting a stable and consistent clustering performance which related to the density of data point in the latent representation.


## Requirements

The required libraries are listed at the top of each script. Please ensure all dependencies are installed before running the code.

## Usage

To run the code, please follow these steps:

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Sinamhr/Code_examples_radiation_paper.git 
    ```
2. Install the required libraries using pip or conda as specified in the headers of each script.
3. Run the example scripts provided to reproduce the results or adapt the code for your own analysis.

## Citation

If you use this code in your work, please cite our paper as follows:

Mehrdad, S., Handorf, D., Höschel, I., Karami, K., Quaas, J., Dipu, S., and Jacobi, C.: Arctic climate response to European radiative forcing: a deep learning study on circulation pattern changes, Weather Clim. Dynam., 5, 1223–1268, https://doi.org/10.5194/wcd-5-1223-2024, 2024.

## License

[MIT License](LICENSE)

## DOI

The code and data associated with this project can be cited using the following DOI:

[10.5281/zenodo.13371085](https://doi.org/10.5281/zenodo.13371085)