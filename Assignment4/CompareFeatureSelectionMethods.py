"""
========================================================
EECS 658 - Assignment 4
Author:    Cameron Denton
Date:      October 21st, 2024
========================================================

Brief Description:
------------------
This program evaluates different feature reduction techniques and 
their impact on classification performance using a Decision Tree 
classifier on the Iris dataset. The feature reduction methods include:

- PCA (Principal Component Analysis)
- Simulated Annealing-based feature selection
- Genetic Algorithm-based feature selection
- Original Feature Space (no reduction)

For each approach, the program reports the selected features, 
confusion matrices, and accuracies. Cross-validation is performed 
using a 50-50 train-test split, evaluated in both directions, and 
the results are combined.

Inputs:
-------
- The Iris dataset is read from "iris.csv".

Outputs:
--------
- Writes results to "output.txt" including:
  - Confusion matrices and accuracies for each method.
  - Selected features after PCA, Simulated Annealing, and Genetic Algorithm.
  - Iteration-by-iteration results for Simulated Annealing.
  - Generation-by-generation results for the Genetic Algorithm.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures.
- Some code and comments suggested by ChatGPT.

========================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA


def load_data():
    """
    --------------------------------------------------------
    Function: load_data
    --------------------------------------------------------
    Description:
    Loads the Iris dataset from 'iris.csv', splits the data into 
    features (X) and labels (y), and encodes class labels as integers.
    
    Inputs:
    - None
    
    Outputs:
    - x: Feature matrix of shape (n_samples, 4)
    - y: Encoded labels of shape (n_samples,)
    - class_names: Array of class label names
    --------------------------------------------------------
    """
    # Load the dataset from CSV
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    # Extract features and labels
    array = dataset.values
    x = array[:, 0:4].astype(float)
    y = array[:, 4]
    
    # Encode labels from strings to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return x, y, label_encoder.classes_


def part1(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f):
    """
    --------------------------------------------------------
    Function: part1
    --------------------------------------------------------
    Description:
    Uses the original feature set (all 4 Iris features) to train 
    and evaluate a Decision Tree classifier. Reports the confusion 
    matrix, accuracy, and the features used.
    
    Inputs:
    - X_Fold1, X_Fold2: Feature subsets for each fold
    - Y_Fold1, Y_Fold2: Label subsets for each fold
    - f: Open file handle to write the results
    
    Outputs:
    - Writes confusion matrix, accuracy, and features used.
    --------------------------------------------------------
    """
    f.write("\nPart 1:\n")
    
    # Original 4 features
    features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
    clf = DecisionTreeClassifier()
    
    # Train on Fold1, test on Fold2
    clf.fit(X_Fold1, Y_Fold1)
    Y_pred1 = clf.predict(X_Fold2)
    
    # Train on Fold2, test on Fold1
    clf.fit(X_Fold2, Y_Fold2)
    Y_pred2 = clf.predict(X_Fold1)
    
    # Combine predictions and actual labels
    Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
    Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
    
    # Compute confusion matrix and accuracy
    cm = confusion_matrix(Y_actual_total, Y_pred_total)
    accuracy = accuracy_score(Y_actual_total, Y_pred_total)
    
    # Write results
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Features used: {features}\n")


def part2(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f):
    """
    --------------------------------------------------------
    Function: part2
    --------------------------------------------------------
    Description:
    Applies PCA for dimensionality reduction. Selects the minimum 
    number of principal components to achieve PoV >= 0.90. Trains 
    and evaluates a Decision Tree classifier on these PCA-reduced 
    features, and reports the eigenvalues, eigenvectors, confusion 
    matrix, accuracy, and PCA features used.
    
    Inputs:
    - X_Fold1, X_Fold2: Feature subsets for each fold
    - Y_Fold1, Y_Fold2: Label subsets for each fold
    - f: Open file handle to write the results
    
    Outputs:
    - Writes eigenvalues, eigenvectors, confusion matrix, accuracy, 
      and selected PCA features.
    --------------------------------------------------------
    """
    f.write("\nPart 2:\n")
    
    # Perform PCA on the combined dataset
    pca = PCA(n_components=4)
    pca.fit(np.vstack((X_Fold1, X_Fold2)))
    
    # Extract eigenvalues and eigenvectors
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    
    f.write("Eigenvalues:\n")
    f.write(f"{eigenvalues}\n")
    f.write("Eigenvectors:\n")
    f.write(f"{eigenvectors}\n")
    
    # Compute PoV (Proportion of Variance)
    PoV = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(PoV >= 0.90) + 1
    selected_PoV = PoV[num_components-1]
    f.write(f"PoV: {selected_PoV}\n")
    f.write(f"Number of components selected: {num_components}\n")
    
    # Transform data to selected components
    Z_Fold1 = pca.transform(X_Fold1)[:, :num_components]
    Z_Fold2 = pca.transform(X_Fold2)[:, :num_components]
    
    clf = DecisionTreeClassifier()
    # Train on Fold1, test on Fold2
    clf.fit(Z_Fold1, Y_Fold1)
    Y_pred1 = clf.predict(Z_Fold2)
    # Train on Fold2, test on Fold1
    clf.fit(Z_Fold2, Y_Fold2)
    Y_pred2 = clf.predict(Z_Fold1)
    
    # Combine predictions and actual labels
    Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
    Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
    
    # Compute confusion matrix and accuracy
    cm = confusion_matrix(Y_actual_total, Y_pred_total)
    accuracy = accuracy_score(Y_actual_total, Y_pred_total)
    
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n")
    f.write(f"Accuracy: {accuracy}\n")
    features = ['z' + str(i+1) for i in range(num_components)]
    f.write(f"Features used: {features}\n")


def part3(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f):
    """
    --------------------------------------------------------
    Function: part3
    --------------------------------------------------------
    Description:
    Uses Simulated Annealing for feature selection from an 
    8-feature set (4 original + 4 PCA-derived). Iteratively 
    perturbs the solution and accepts it based on probability 
    to escape local optima. Prints details for each iteration 
    (feature subset, accuracy, acceptance probability, etc.) and 
    reports the best solution found along with its confusion matrix 
    and accuracy.
    
    Inputs:
    - X_Fold1, X_Fold2: Feature subsets for each fold
    - Y_Fold1, Y_Fold2: Label subsets for each fold
    - f: Open file handle to write the results
    
    Outputs:
    - Iteration-by-iteration details (feature subsets, accuracy, etc.)
    - Best solution and its confusion matrix and accuracy.
    --------------------------------------------------------
    """
    f.write("\nPart 3:\n")
    
    # Perform PCA on combined dataset for 4 additional features
    pca = PCA(n_components=4)
    X_combined = np.vstack((X_Fold1, X_Fold2))
    pca.fit(X_combined)
    Z_combined = pca.transform(X_combined)
    
    # Split PCA-transformed data back into folds
    Z_Fold1 = Z_combined[:len(X_Fold1), :]
    Z_Fold2 = Z_combined[len(X_Fold1):, :]
    
    # Combine original + PCA features (8 features total)
    features_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'z1', 'z2', 'z3', 'z4']
    X_Fold1_8 = np.hstack((X_Fold1, Z_Fold1))
    X_Fold2_8 = np.hstack((X_Fold2, Z_Fold2))
    
    # Simulated Annealing parameters
    iterations = 100
    c = 1
    restart_value = 10
    n_features = 8
    
    # Initialize solution randomly
    current_solution = np.random.choice([0, 1], size=n_features)
    if np.sum(current_solution) == 0:
        current_solution[np.random.randint(0, n_features)] = 1
    
    # Evaluate the initial solution
    clf = DecisionTreeClassifier()
    selected_features = np.where(current_solution == 1)[0]
    clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
    Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
    clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
    Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
    Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
    Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
    current_accuracy = accuracy_score(Y_actual_total, Y_pred_total)
    
    best_accuracy = current_accuracy
    best_solution = current_solution.copy()
    no_improvement_count = 0
    
    # Simulated Annealing loop
    for iteration in range(iterations):
        # Perturb the solution (flip 1 or 2 bits)
        new_solution = current_solution.copy()
        num_perturb = np.random.randint(1, 3)  # flip 1 or 2 bits
        flip_indices = np.random.choice(n_features, size=num_perturb, replace=False)
        new_solution[flip_indices] = 1 - new_solution[flip_indices]
        if np.sum(new_solution) == 0:
            new_solution[np.random.randint(0, n_features)] = 1
        
        # Evaluate new solution
        selected_features = np.where(new_solution == 1)[0]
        clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
        Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
        clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
        Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
        Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
        Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
        new_accuracy = accuracy_score(Y_actual_total, Y_pred_total)
        
        delta = new_accuracy - current_accuracy
        if delta > 0:
            # Improved solution
            pr_accept = 1
            status = 'Improved'
            current_solution = new_solution
            current_accuracy = new_accuracy
            if new_accuracy > best_accuracy:
                best_accuracy = new_accuracy
                best_solution = new_solution.copy()
            no_improvement_count = 0
            rand_uniform = 'N/A'
        else:
            # Possibly accept worse solution
            pr_accept = np.exp(c * delta)
            rand_uniform = np.random.uniform()
            if pr_accept > rand_uniform:
                status = 'Accepted'
                current_solution = new_solution
                current_accuracy = new_accuracy
            else:
                status = 'Discarded'
            no_improvement_count += 1
        
        # Restart if no improvement for a while
        if no_improvement_count >= restart_value:
            current_solution = np.random.choice([0,1], size=n_features)
            if np.sum(current_solution) == 0:
                current_solution[np.random.randint(0, n_features)] = 1
            selected_features = np.where(current_solution == 1)[0]
            clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
            Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
            clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
            Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
            Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
            Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
            current_accuracy = accuracy_score(Y_actual_total, Y_pred_total)
            no_improvement_count = 0
            status = 'Restart'
            pr_accept = 'N/A'
            rand_uniform = 'N/A'
        
        selected_feature_names = [features_names[i] for i in np.where(current_solution ==1)[0]]
        f.write(f"Iteration {iteration+1}:\n")
        f.write(f"Subset of features: {selected_feature_names}\n")
        f.write(f"Accuracy: {current_accuracy}\n")
        f.write(f"Pr[accept]: {pr_accept}\n")
        f.write(f"Random Uniform: {rand_uniform}\n")
        f.write(f"Status: {status}\n\n")
    
    # After all iterations, evaluate best solution
    f.write("Best solution found:\n")
    selected_feature_names = [features_names[i] for i in np.where(best_solution ==1)[0]]
    f.write(f"Subset of features: {selected_feature_names}\n")
    f.write(f"Accuracy: {best_accuracy}\n")
    
    selected_features = np.where(best_solution ==1)[0]
    clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
    Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
    clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
    Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
    Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
    Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
    cm = confusion_matrix(Y_actual_total, Y_pred_total)
    accuracy = accuracy_score(Y_actual_total, Y_pred_total)
    
    f.write("Final Confusion Matrix:\n")
    f.write(f"{cm}\n")
    f.write(f"Final Accuracy: {accuracy}\n")
    f.write(f"Features used: {selected_feature_names}\n")


def part4(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f):
    """
    --------------------------------------------------------
    Function: part4
    --------------------------------------------------------
    Description:
    Uses a Genetic Algorithm (GA) to select features from the 
    8-feature set (4 original + 4 PCA-derived). Over 50 generations, 
    it evolves a population, applies crossover and mutation, and 
    selects the best solutions. Writes details for each generation 
    and finally reports the best solution, confusion matrix, and 
    accuracy.
    
    Inputs:
    - X_Fold1, X_Fold2: Feature subsets for each fold
    - Y_Fold1, Y_Fold2: Label subsets for each fold
    - f: Open file handle to write the results
    
    Outputs:
    - Generation-by-generation details of individuals and accuracies.
    - Final best individual, confusion matrix, and accuracy.
    --------------------------------------------------------
    """
    f.write("\nPart 4:\n")
    
    # Perform PCA on combined dataset for 4 additional features
    pca = PCA(n_components=4)
    X_combined = np.vstack((X_Fold1, X_Fold2))
    pca.fit(X_combined)
    Z_combined = pca.transform(X_combined)
    
    # Split PCA-transformed data back into folds
    Z_Fold1 = Z_combined[:len(X_Fold1), :]
    Z_Fold2 = Z_combined[len(X_Fold1):, :]
    
    # Combine original + PCA features (8 features total)
    features_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'z1', 'z2', 'z3', 'z4']
    X_Fold1_8 = np.hstack((X_Fold1, Z_Fold1))
    X_Fold2_8 = np.hstack((X_Fold2, Z_Fold2))
    
    # Initial population (5 individuals)
    initial_population = [
        ['z1', 'sepal-length', 'sepal-width', 'petal-length', 'petal-width'],
        ['z1', 'z2', 'sepal-width', 'petal-length', 'petal-width'],
        ['z1', 'z2', 'z3', 'sepal-width', 'petal-length'],
        ['z1', 'z2', 'z3', 'z4', 'sepal-width'],
        ['z1', 'z2', 'z3', 'z4', 'sepal-length']
    ]
    
    population_size = 5
    n_features = 8
    max_generations = 50
    mutation_rate = 0.1
    
    # Convert initial population to binary vectors
    population = []
    for individual in initial_population:
        individual_vector = np.zeros(n_features, dtype=int)
        for feature in individual:
            index = features_names.index(feature)
            individual_vector[index] = 1
        population.append(individual_vector)
    
    best_individual = None
    best_accuracy = 0
    
    # Genetic Algorithm loop
    for generation in range(max_generations):
        fitness_scores = []
        for individual in population:
            selected_features = np.where(individual ==1)[0]
            if len(selected_features) ==0:
                fitness_scores.append(0)
                continue
            clf = DecisionTreeClassifier()
            # Train on Fold1, test on Fold2
            clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
            Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
            # Train on Fold2, test on Fold1
            clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
            Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
            Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
            Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
            accuracy = accuracy_score(Y_actual_total, Y_pred_total)
            fitness_scores.append(accuracy)
        
        # Sort individuals by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        
        # Update best individual
        if fitness_scores[0] > best_accuracy:
            best_accuracy = fitness_scores[0]
            best_individual = population[0].copy()
        
        # Write top individuals of this generation
        f.write(f"Generation {generation+1}:\n")
        for i in range(min(5, len(population))):
            individual = population[i]
            acc = fitness_scores[i]
            selected_feature_names = [features_names[j] for j in np.where(individual ==1)[0]]
            f.write(f"Individual {i+1}:\n")
            f.write(f"Features: {selected_feature_names}\n")
            f.write(f"Accuracy: {acc}\n")
        f.write("\n")
        
        # Create next generation
        next_generation = population[:2]  # Elitism: keep top 2
        while len(next_generation) < population_size:
            # Selection by roulette wheel
            total_fitness = sum(fitness_scores)
            if total_fitness ==0:
                probabilities = [1/len(population)] * len(population)
            else:
                probabilities = [f/total_fitness for f in fitness_scores]
            parents_indices = np.random.choice(len(population), size=2, replace=False, p=probabilities)
            parent1 = population[parents_indices[0]]
            parent2 = population[parents_indices[1]]
            
            # Crossover
            crossover_point = np.random.randint(1, n_features-1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            for i in range(n_features):
                if np.random.rand() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Ensure at least one feature is selected
            if np.sum(child) ==0:
                child[np.random.randint(0, n_features)] =1
            next_generation.append(child)
        
        population = next_generation
    
    # After final generation, print final population details
    f.write("Final Generation Results:\n")
    for i in range(min(5, len(population))):
        individual = population[i]
        selected_features = np.where(individual ==1)[0]
        if len(selected_features) ==0:
            continue
        clf = DecisionTreeClassifier()
        clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
        Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
        clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
        Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
        Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
        Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
        accuracy = accuracy_score(Y_actual_total, Y_pred_total)
        selected_feature_names = [features_names[j] for j in selected_features]
        f.write(f"Individual {i+1}:\n")
        f.write(f"Features: {selected_feature_names}\n")
        f.write(f"Accuracy: {accuracy}\n")
    
    # Evaluate best individual
    selected_features = np.where(best_individual ==1)[0]
    clf = DecisionTreeClassifier()
    clf.fit(X_Fold1_8[:, selected_features], Y_Fold1)
    Y_pred1 = clf.predict(X_Fold2_8[:, selected_features])
    clf.fit(X_Fold2_8[:, selected_features], Y_Fold2)
    Y_pred2 = clf.predict(X_Fold1_8[:, selected_features])
    Y_pred_total = np.concatenate((Y_pred1, Y_pred2))
    Y_actual_total = np.concatenate((Y_Fold2, Y_Fold1))
    cm = confusion_matrix(Y_actual_total, Y_pred_total)
    accuracy = accuracy_score(Y_actual_total, Y_pred_total)
    selected_feature_names = [features_names[j] for j in selected_features]
    
    f.write("\nBest Individual:\n")
    f.write(f"Features: {selected_feature_names}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{cm}\n")
    f.write(f"Accuracy: {accuracy}\n")


def main():
    """
    --------------------------------------------------------
    Function: main
    --------------------------------------------------------
    Description:
    Loads the Iris dataset, splits it into two folds, and runs:
    - Part 1: Original features
    - Part 2: PCA-based feature reduction
    - Part 3: Simulated Annealing for feature selection
    - Part 4: Genetic Algorithm for feature selection
    
    All results are written to 'output.txt'.
    
    Inputs:
    - None
    
    Outputs:
    - Writes results to output.txt for all parts.
    --------------------------------------------------------
    """
    # Load data
    x, y, class_names = load_data()
    
    # 50-50 train-test split
    X_Fold1, X_Fold2, Y_Fold1, Y_Fold2 = train_test_split(x, y, test_size=0.5, random_state=1)
    
    with open('output.txt', 'w') as f:
        part1(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f)
        part2(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f)
        part3(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f)
        part4(X_Fold1, X_Fold2, Y_Fold1, Y_Fold2, f)


if __name__ == "__main__":
    main()
