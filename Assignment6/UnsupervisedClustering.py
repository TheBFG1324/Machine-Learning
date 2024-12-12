"""
========================================================
EECS 658 - Assignment 6
Author:    Cameron Denton
Date:      November 12th, 2024
========================================================

Brief Description:
------------------
This program applies three different unsupervised learning clustering algorithms 
on the Iris dataset and visualizes their performance and metrics:

1. k-means Clustering
   - Runs k-means for k = 1 to 20 clusters
   - Plots Reconstruction Error vs k
   - Allows the user to input an "elbow_k" to determine the best number of clusters
   - Prints confusion matrix and accuracy if the chosen number of clusters matches 
     the true number of classes (k=3 for Iris)

2. Gaussian Mixture Models (GMM)
   - Runs GMM for k = 1 to 20 clusters
   - Plots AIC vs k and BIC vs k
   - Allows the user to input "aic_elbow_k" and "bic_elbow_k" based on the plots
   - Prints confusion matrix and accuracy if the chosen cluster count matches the 
     true number of classes (k=3 for Iris)
     
3. Self-Organizing Map (SOM)
   - Evaluates SOM with different grid sizes (3x3, 7x7, 15x15, 25x25)
   - Trains each SOM and prints the quantization error
   - Plots the U-Matrix and shows the placement of Iris samples on the SOM
   - Plots the quantization errors for different grid sizes

Inputs:
-------
- The Iris dataset is read from "iris.csv".
- User inputs for elbow points on k, AIC, and BIC plots.

Outputs:
--------
- Plots:
  - k-means: Reconstruction Error vs k
  - GMM: AIC vs k, BIC vs k
  - SOM: U-Matrices for different grid sizes, Quantization Error vs Grid Size
- Confusion matrices and accuracy for chosen cluster solutions (when possible).

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures and assignment repository.
- Some code and comments suggested by ChatGPT.

========================================================
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from minisom import MiniSom

def load_data():
    """
    --------------------------------------------------------
    Function: load_data
    --------------------------------------------------------
    Description:
    Loads the Iris dataset from 'iris.csv' into a pandas DataFrame.
    
    Inputs:
    - None
    
    Outputs:
    - dataset: A pandas DataFrame containing the Iris dataset.
    --------------------------------------------------------
    """
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    return dataset

def plot_graph(arr, name):
    """
    --------------------------------------------------------
    Function: plot_graph
    --------------------------------------------------------
    Description:
    Plots a given array of values (arr) against k=1 to 20 with the 
    specified name as the Y-axis label and the title.
    
    Inputs:
    - arr: A list or array of values to plot.
    - name: A string used as the label for the Y-axis and title suffix.
    
    Outputs:
    - Displays a line plot with points, labeled axes, and title.
    --------------------------------------------------------
    """
    plt.figure()
    plt.plot(range(1, 21), arr, marker='o')
    plt.title(f'{name} vs. k')
    plt.xlabel('k')
    plt.xticks(np.arange(1, 21, 1))
    plt.ylabel(name)
    plt.show()

def map_labels(true_labels, pred_labels):
    """
    --------------------------------------------------------
    Function: map_labels
    --------------------------------------------------------
    Description:
    Maps cluster labels to the true labels for evaluation. Uses the 
    Hungarian algorithm (linear_sum_assignment) on the confusion 
    matrix to maximize label alignment.
    
    Inputs:
    - true_labels: Ground truth class labels.
    - pred_labels: Predicted cluster labels from an unsupervised model.
    
    Outputs:
    - mapped_pred_labels: Predicted labels mapped to best-match true labels.
    --------------------------------------------------------
    """
    cost_matrix = -confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_mapping = dict(zip(col_ind, row_ind))
    mapped_pred_labels = np.array([label_mapping[label] for label in pred_labels])
    return mapped_pred_labels

def part1_kmeans(X, y):
    """
    --------------------------------------------------------
    Function: part1_kmeans
    --------------------------------------------------------
    Description:
    Runs k-means clustering for k=1 to 20, plotting reconstruction 
    error (inertia) against k. Prompts the user to input an "elbow_k".
    If elbow_k=3 (the true number of Iris classes), prints confusion 
    matrix and accuracy. Also prints results specifically for k=3.
    
    Inputs:
    - X: Feature matrix
    - y: True class labels (encoded)
    
    Outputs:
    - Plots Reconstruction Error vs k
    - Prompts for elbow_k and prints confusion matrix and accuracy.
    --------------------------------------------------------
    """
    # Compute reconstruction errors for k-means
    reconstruction_errors = []
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        reconstruction_errors.append(kmeans.inertia_)

    # Plot reconstruction error vs k
    plot_graph(reconstruction_errors, 'Reconstruction Error')

    # User input for elbow_k
    elbow_k = int(input("Enter the elbow_k value determined from the plot (e.g., 3): "))

    # Run k-means with elbow_k
    kmeans = KMeans(n_clusters=elbow_k, random_state=42)
    kmeans.fit(X)
    predicted_labels = kmeans.predict(X)

    if elbow_k == 3:
        mapped_labels = map_labels(y, predicted_labels)
        conf_matrix = confusion_matrix(y, mapped_labels)
        accuracy = accuracy_score(y, mapped_labels)
        print("\nConfusion Matrix (k = elbow_k):")
        print(conf_matrix)
        print(f"Accuracy (k = elbow_k): {accuracy:.2f}")
    else:
        print("\nCannot calculate Accuracy Score because the chosen number of clusters doesn't match 3.")
        print("Confusion Matrix (k = elbow_k):")
        print(confusion_matrix(y, predicted_labels))

    # Also show results for k=3
    kmeans_3 = KMeans(n_clusters=3, random_state=42)
    kmeans_3.fit(X)
    predicted_labels_3 = kmeans_3.predict(X)
    mapped_labels_3 = map_labels(y, predicted_labels_3)
    conf_matrix_3 = confusion_matrix(y, mapped_labels_3)
    accuracy_3 = accuracy_score(y, mapped_labels_3)
    print("\nConfusion Matrix (k = 3):")
    print(conf_matrix_3)
    print(f"Accuracy (k = 3): {accuracy_3:.2f}")

def part2_gmm(X, y):
    """
    --------------------------------------------------------
    Function: part2_gmm
    --------------------------------------------------------
    Description:
    Runs Gaussian Mixture Models (GMM) for k=1 to 20. Plots AIC and BIC 
    vs k. Prompts user for 'aic_elbow_k' and 'bic_elbow_k'. Prints 
    confusion matrix and accuracy if chosen k=3 matches the number of classes.
    
    Inputs:
    - X: Feature matrix
    - y: True class labels (encoded)
    
    Outputs:
    - Plots AIC vs k, BIC vs k
    - Confusion matrices and accuracies for chosen k values.
    --------------------------------------------------------
    """
    # Compute AIC for k=1 to 20
    aic_values = []
    for k in range(1, 21):
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        gmm.fit(X)
        aic_values.append(gmm.aic(X))

    # Plot AIC vs k
    plot_graph(aic_values, 'AIC')
    aic_elbow_k = int(input("Enter the aic_elbow_k value determined from the AIC plot (e.g., 3): "))

    # Compute BIC for k=1 to 20
    bic_values = []
    for k in range(1, 21):
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        gmm.fit(X)
        bic_values.append(gmm.bic(X))

    # Plot BIC vs k
    plot_graph(bic_values, 'BIC')
    bic_elbow_k = int(input("Enter the bic_elbow_k value determined from the BIC plot (e.g., 3): "))

    # Evaluate GMM at aic_elbow_k
    gmm_aic = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag', random_state=42)
    gmm_aic.fit(X)
    predicted_labels_aic = gmm_aic.predict(X)
    if aic_elbow_k == 3:
        mapped_labels_aic = map_labels(y, predicted_labels_aic)
        conf_matrix_aic = confusion_matrix(y, mapped_labels_aic)
        accuracy_aic = accuracy_score(y, mapped_labels_aic)
        print("\nConfusion Matrix (k = aic_elbow_k):")
        print(conf_matrix_aic)
        print(f"Accuracy (k = aic_elbow_k): {accuracy_aic:.2f}")
    else:
        print("\nCannot calculate Accuracy Score for aic_elbow_k as it is not 3.")
        print("Confusion Matrix (k = aic_elbow_k):")
        print(confusion_matrix(y, predicted_labels_aic))

    # Evaluate GMM at bic_elbow_k
    gmm_bic = GaussianMixture(n_components=bic_elbow_k, covariance_type='diag', random_state=42)
    gmm_bic.fit(X)
    predicted_labels_bic = gmm_bic.predict(X)
    if bic_elbow_k == 3:
        mapped_labels_bic = map_labels(y, predicted_labels_bic)
        conf_matrix_bic = confusion_matrix(y, mapped_labels_bic)
        accuracy_bic = accuracy_score(y, mapped_labels_bic)
        print("\nConfusion Matrix (k = bic_elbow_k):")
        print(conf_matrix_bic)
        print(f"Accuracy (k = bic_elbow_k): {accuracy_bic:.2f}")
    else:
        print("\nCannot calculate Accuracy Score for bic_elbow_k as it is not 3.")
        print("Confusion Matrix (k = bic_elbow_k):")
        print(confusion_matrix(y, predicted_labels_bic))

def part3_som(X, y):
    """
    --------------------------------------------------------
    Function: part3_som
    --------------------------------------------------------
    Description:
    Evaluates Self-Organizing Map (SOM) performance for different 
    grid sizes. Computes quantization error for each grid and 
    plots the resulting U-Matrices. Finally, plots quantization 
    error vs grid size.
    
    Grid sizes: [3x3, 7x7, 15x15, 25x25]
    
    Inputs:
    - X: Feature matrix
    - y: True class labels (encoded)
    
    Outputs:
    - Prints quantization errors for each grid size
    - Displays SOM U-Matrix plots with data points
    - Plots quantization error vs grid size
    --------------------------------------------------------
    """
    # Normalize features
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)

    grid_sizes = [3, 7, 15, 25]
    quantization_errors = []

    for size in grid_sizes:
        som = MiniSom(x=size, y=size, input_len=X_norm.shape[1], sigma=0.3, learning_rate=0.5, random_seed=42)
        som.random_weights_init(X_norm)
        som.train_random(data=X_norm, num_iteration=1000)

        # Compute quantization error
        q_error = som.quantization_error(X_norm)
        quantization_errors.append(q_error)
        print(f"\nQuantization Error for grid size {size}x{size}: {q_error:.4f}")

        # Plot U-Matrix
        plt.figure(figsize=(7, 7))
        plt.pcolor(som.distance_map().T, cmap='bone_r')
        plt.colorbar()
        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for cnt, xx in enumerate(X_norm):
            w = som.winner(xx)
            plt.plot(w[0] + 0.5, w[1] + 0.5, markers[y[cnt]], markerfacecolor='None',
                     markeredgecolor=colors[y[cnt]], markersize=12, markeredgewidth=2)
        plt.title(f'SOM Response Map for grid size {size}x{size}')
        plt.show()

    # Plot quantization error vs grid sizes
    plt.figure()
    plt.plot(grid_sizes, quantization_errors, marker='o')
    plt.title('Quantization Error vs Grid Size')
    plt.xlabel('Grid Size (n x n)')
    plt.ylabel('Quantization Error')
    plt.show()


if __name__ == '__main__':
    data = load_data()
    X = data[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].values
    y = LabelEncoder().fit_transform(data['class'].values)  # Encode target labels

    print("Part 1: k-Means Clustering")
    part1_kmeans(X, y)

    print("\nPart 2: Gaussian Mixture Models (GMM)")
    part2_gmm(X, y)

    print("\nPart 3: Self Organizing Map (SOM)")
    part3_som(X, y)
