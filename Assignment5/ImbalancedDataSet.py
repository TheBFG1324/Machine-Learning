"""
========================================================
EECS 658 - Assignment 5
Author:    Cameron Denton
Date:      October 29th, 2024
========================================================

Brief Description:
------------------
This program demonstrates the impact of various resampling techniques 
on an imbalanced dataset, using a Multi-Layer Perceptron (MLP) classifier 
for evaluation. The dataset is first split into two folds (using 
StratifiedKFold for maintaining class proportion). The MLP is trained 
and evaluated in both train/test configurations, and the results are 
combined. Different resampling methods are applied to the training data, 
and their results are compared in terms of confusion matrices and accuracies.

The following scenarios are evaluated:
- Given Imbalanced Data (no resampling)
- Random Oversampling
- SMOTE Oversampling
- ADASYN Oversampling
- Random Undersampling
- Cluster Undersampling
- Tomek Links Undersampling

Inputs:
-------
- "imbalanced-iris.csv" dataset provided by the instructor.

Outputs:
--------
- For each scenario, the confusion matrix and accuracy are printed.
- For the given imbalanced data scenario, additional metrics such as 
  Class Balanced Accuracy and Balanced Accuracy are also computed.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures.
- Some code and comments provided by ChatGPT.

========================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks


def load_data():
    """
    --------------------------------------------------------
    Function: load_data
    --------------------------------------------------------
    Description:
    Loads the imbalanced Iris dataset from "imbalanced-iris.csv", 
    extracts features (X) and labels (y), and encodes the class 
    labels into numerical form.
    
    Inputs:
    - None
    
    Outputs:
    - x: Feature matrix (float)
    - y: Encoded label vector (int)
    - class_names: Array of class names corresponding to the encoded labels
    --------------------------------------------------------
    """
    # Load dataset from CSV
    url = "imbalanced-iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    # Split dataset into features (X) and labels (y)
    array = dataset.values
    x = array[:, 0:4].astype(float)
    y = array[:, 4]
    
    # Encode string labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return x, y, label_encoder.classes_


def part1(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2):
    """
    --------------------------------------------------------
    Function: part1
    --------------------------------------------------------
    Description:
    Evaluates the MLP model on the original (imbalanced) dataset 
    without any resampling. It computes and prints:
    - Confusion Matrix
    - Accuracy
    - Class Balanced Accuracy
    - Balanced Accuracy (computed manually)
    - Balanced Accuracy (from scikit-learn)
    
    Inputs:
    - X_train1, X_test1, Y_train1, Y_test1: First fold train/test
    - X_train2, X_test2, Y_train2, Y_test2: Second fold train/test
    
    Outputs:
    - Prints the metrics described above.
    --------------------------------------------------------
    """
    print("Part 1:\n")
    
    # Initialize MLP model
    model = MLPClassifier(random_state=1, max_iter=1000)
    
    # Train on first fold and predict on the second
    model.fit(X_train1, Y_train1)
    pred1 = model.predict(X_test1)
    
    # Train on second fold and predict on the first
    model.fit(X_train2, Y_train2)
    pred2 = model.predict(X_test2)
    
    # Combine actual and predicted labels from both folds
    actual = np.concatenate([Y_test1, Y_test2])
    predicted = np.concatenate([pred1, pred2])
    
    # Compute confusion matrix and accuracy
    cm = confusion_matrix(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    
    print("Confusion Matrix:")
    print(cm)
    print("\nAccuracy Score:", accuracy)
    
    # Calculate Class Balanced Accuracy (CBA)
    # CBA: For each class i, calculate Accuracy_i = (TP_i + TN_i) / Total_samples
    # Then average over all classes.
    num_classes = cm.shape[0]
    total_samples = cm.sum()
    class_accuracy_list = []
    for i in range(num_classes):
        TP_i = cm[i, i]
        FN_i = cm[i, :].sum() - TP_i
        FP_i = cm[:, i].sum() - TP_i
        TN_i = total_samples - (TP_i + FN_i + FP_i)
        class_accuracy_i = (TP_i + TN_i) / total_samples
        class_accuracy_list.append(class_accuracy_i)
    class_balanced_accuracy = sum(class_accuracy_list) / num_classes
    print("\nClass Balanced Accuracy:", class_balanced_accuracy)
    
    # Calculate Balanced Accuracy
    # Balanced accuracy is the average of sensitivity and specificity for each class,
    # then averaged over all classes.
    sensitivity_list = []
    specificity_list = []
    for i in range(num_classes):
        TP_i = cm[i, i]
        FN_i = cm[i, :].sum() - TP_i
        FP_i = cm[:, i].sum() - TP_i
        TN_i = total_samples - (TP_i + FP_i + FN_i)
        
        sensitivity_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) != 0 else 0
        specificity_i = TN_i / (TN_i + FP_i) if (TN_i + FP_i) != 0 else 0
        sensitivity_list.append(sensitivity_i)
        specificity_list.append(specificity_i)
    
    balanced_accuracy = sum([(s + sp) / 2 for s, sp in zip(sensitivity_list, specificity_list)]) / num_classes
    print("\nBalanced Accuracy:", balanced_accuracy)
    
    # Balanced Accuracy from scikit-learn
    skl_balanced_accuracy = balanced_accuracy_score(actual, predicted)
    print("\nScikit-learn Balanced Accuracy Score:", skl_balanced_accuracy)
    print("\n" + "-"*50 + "\n")


def part2(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2):
    """
    --------------------------------------------------------
    Function: part2
    --------------------------------------------------------
    Description:
    Applies three oversampling techniques on the training data:
    - Random Oversampling
    - SMOTE Oversampling
    - ADASYN Oversampling
    
    For each technique, an MLP model is trained using the oversampled 
    data and evaluated. Prints the confusion matrix and accuracy 
    for each method.
    
    Inputs:
    - X_train1, X_test1, Y_train1, Y_test1: First fold train/test
    - X_train2, X_test2, Y_train2, Y_test2: Second fold train/test
    
    Outputs:
    - Prints confusion matrices and accuracies for each oversampling method.
    --------------------------------------------------------
    """
    print("Part 2:\n")
    
    # Define oversampling methods
    oversampling_methods = {
        'Random Oversampling': RandomOverSampler(),
        'SMOTE Oversampling': SMOTE(),
        'ADASYN Oversampling': ADASYN(sampling_strategy='minority')
    }
    
    # Evaluate each oversampling technique
    for method_name, sampler in oversampling_methods.items():
        print(f"{method_name}:\n")
        
        # Initialize MLP model
        model = MLPClassifier(random_state=1, max_iter=1000)
        
        # First fold (train on resampled data, test on original test)
        X_resampled1, Y_resampled1 = sampler.fit_resample(X_train1, Y_train1)
        model.fit(X_resampled1, Y_resampled1)
        pred1 = model.predict(X_test1)
        
        # Second fold
        X_resampled2, Y_resampled2 = sampler.fit_resample(X_train2, Y_train2)
        model.fit(X_resampled2, Y_resampled2)
        pred2 = model.predict(X_test2)
        
        # Combine actual and predicted labels
        actual = np.concatenate([Y_test1, Y_test2])
        predicted = np.concatenate([pred1, pred2])
        
        # Compute confusion matrix and accuracy
        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        
        print("Confusion Matrix:")
        print(cm)
        print("\nAccuracy Score:", accuracy)
        print("\n" + "-"*50 + "\n")


def part3(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2):
    """
    --------------------------------------------------------
    Function: part3
    --------------------------------------------------------
    Description:
    Applies three undersampling techniques on the training data:
    - Random Undersampling
    - Cluster Centroids (Cluster Undersampling)
    - Tomek Links Undersampling
    
    For each technique, an MLP model is trained using the undersampled 
    data and evaluated. Prints the confusion matrix and accuracy 
    for each method.
    
    Inputs:
    - X_train1, X_test1, Y_train1, Y_test1: First fold train/test
    - X_train2, X_test2, Y_train2, Y_test2: Second fold train/test
    
    Outputs:
    - Prints confusion matrices and accuracies for each undersampling method.
    --------------------------------------------------------
    """
    print("Part 3:\n")
    
    # Define undersampling methods
    undersampling_methods = {
        'Random Undersampling': RandomUnderSampler(),
        'Cluster Undersampling': ClusterCentroids(),
        'Tomek Links Undersampling': TomekLinks()
    }
    
    # Evaluate each undersampling technique
    for method_name, sampler in undersampling_methods.items():
        print(f"{method_name}:\n")
        
        # Initialize MLP model
        model = MLPClassifier(random_state=1, max_iter=1000)
        
        # First fold (train on resampled data, test on original test)
        X_resampled1, Y_resampled1 = sampler.fit_resample(X_train1, Y_train1)
        model.fit(X_resampled1, Y_resampled1)
        pred1 = model.predict(X_test1)
        
        # Second fold
        X_resampled2, Y_resampled2 = sampler.fit_resample(X_train2, Y_train2)
        model.fit(X_resampled2, Y_resampled2)
        pred2 = model.predict(X_test2)
        
        # Combine actual and predicted labels
        actual = np.concatenate([Y_test1, Y_test2])
        predicted = np.concatenate([pred1, pred2])
        
        # Compute confusion matrix and accuracy
        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        
        print("Confusion Matrix:")
        print(cm)
        print("\nAccuracy Score:", accuracy)
        print("\n" + "-"*50 + "\n")


def main():
    """
    --------------------------------------------------------
    Function: main
    --------------------------------------------------------
    Description:
    Loads the imbalanced Iris dataset, splits it into two folds 
    using StratifiedKFold, and then runs the three parts:
    
    - Part 1: Evaluate MLP on original imbalanced data.
    - Part 2: Evaluate MLP after applying oversampling methods.
    - Part 3: Evaluate MLP after applying undersampling methods.
    
    Prints the confusion matrices and accuracies for each scenario.
    
    Inputs:
    - None
    
    Outputs:
    - Prints results to stdout.
    --------------------------------------------------------
    """
    # Load data
    x, y, class_names = load_data()
    
    # Use StratifiedKFold to maintain class proportions in the folds
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    splits = list(skf.split(x, y))
    train_index1, test_index1 = splits[0]
    train_index2, test_index2 = splits[1]

    # Prepare train/test sets for both folds
    X_train1, Y_train1 = x[train_index1], y[train_index1]
    X_test1, Y_test1 = x[test_index1], y[test_index1]

    X_train2, Y_train2 = x[train_index2], y[train_index2]
    X_test2, Y_test2 = x[test_index2], y[test_index2]
    
    # Run parts
    part1(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2)
    part2(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2)
    part3(X_train1, X_test1, Y_train1, Y_test1, X_train2, X_test2, Y_train2, Y_test2)


if __name__ == '__main__':
    main()
