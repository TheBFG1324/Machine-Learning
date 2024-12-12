"""
========================================================
EECS 658 - Assignment 3
Author:    Cameron Denton
Date:      September 30th, 2024
========================================================

Brief Description:
------------------
This program evaluates and compares the performance of a variety of 
machine learning models on the Iris dataset. The models include:

- Linear Regression (LinearRegression)
- Polynomial Regression (degree 2) (LinearRegression)
- Polynomial Regression (degree 3) (LinearRegression)
- Naive Bayes (GaussianNB)
- k-Nearest Neighbors (KNeighborsClassifier)
- Linear Discriminant Analysis (LinearDiscriminantAnalysis)
- Quadratic Discriminant Analysis (QuadraticDiscriminantAnalysis)
- Support Vector Machine (svm.LinearSVC)
- Decision Tree (DecisionTreeClassifier)
- Random Forest (RandomForestClassifier)
- ExtraTrees (ExtraTreesClassifier)
- Neural Network (MLPClassifier)

For each model, the dataset is split into two folds (50%-50%). The model 
is trained on one fold and tested on the other, and then vice versa. 
The results from both folds are combined to compute overall accuracy 
and confusion matrices.

Inputs:
-------
- Iris dataset loaded from "iris.csv"

Outputs:
--------
- Prints out the confusion matrix and accuracy for each model evaluated.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures.
- Some code and comments suggested by ChatGPT.

========================================================
"""

# ========================================================
# Import required libraries and models
# ========================================================
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


def load_data():
    """
    --------------------------------------------------------
    Function: load_data
    --------------------------------------------------------
    Description:
    Loads and formats the Iris dataset from 'iris.csv'. Extracts 
    features (X) and target labels (y), and encodes class labels 
    from strings to integers.
    
    Inputs:
    - None
    
    Outputs:
    - X: Feature matrix of shape (n_samples, n_features)
    - y: Encoded label vector of shape (n_samples,)
    - class_names: Array of class names corresponding to encoded labels
    --------------------------------------------------------
    """
    # Specify data file and column names
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    # Extract feature columns and label column
    array = dataset.values
    X = array[:, 0:4].astype(float)
    y = array[:, 4]
    
    # Encode string labels into numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder.classes_


def main():
    """
    --------------------------------------------------------
    Function: main
    --------------------------------------------------------
    Description:
    Loads the Iris dataset, splits it into two equal folds, and 
    evaluates various ML models on both folds. For polynomial 
    regression models, the feature sets are transformed before 
    training and prediction. Results from both folds are combined 
    to produce overall accuracy and confusion matrices for each 
    model.
    
    Inputs:
    - None
    
    Outputs:
    - Prints the confusion matrix and accuracy for each model evaluated.
    --------------------------------------------------------
    """
    # Load the dataset
    X, y, class_names = load_data()
    
    # Perform a 50-50 split to create two folds
    X_Fold1, X_Fold2, Y_Fold1, Y_Fold2 = train_test_split(X, y, test_size=0.5, random_state=1)
    
    # Define the models to be evaluated
    models = [
        ('Linear Regression', LinearRegression()),
        ('Polynomial Degree 2 Regression', LinearRegression()),
        ('Polynomial Degree 3 Regression', LinearRegression()),
        ('Naive Bayes', GaussianNB()),
        ('kNN', KNeighborsClassifier()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
        ('SVM', svm.LinearSVC(max_iter=10000)),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier()),
        ('ExtraTrees', ExtraTreesClassifier()),
        ('Neural Network', MLPClassifier(max_iter=1000))
    ]
    
    print("========================================================")
    print("Model Evaluation Results")
    print("========================================================\n")
    
    # Evaluate each model
    for name, model in models:
        # Training/testing splits for each fold
        X_train1, Y_train1, X_test1, Y_test1 = X_Fold1, Y_Fold1, X_Fold2, Y_Fold2
        X_train2, Y_train2, X_test2, Y_test2 = X_Fold2, Y_Fold2, X_Fold1, Y_Fold1
        
        # Handle polynomial transformations for polynomial regression models
        if 'Polynomial Degree 2' in name:
            poly = PolynomialFeatures(degree=2)
            X_train1, X_test1 = poly.fit_transform(X_train1), poly.transform(X_test1)
            X_train2, X_test2 = poly.fit_transform(X_train2), poly.transform(X_test2)
        
        elif 'Polynomial Degree 3' in name:
            poly = PolynomialFeatures(degree=3)
            X_train1, X_test1 = poly.fit_transform(X_train1), poly.transform(X_test1)
            X_train2, X_test2 = poly.fit_transform(X_train2), poly.transform(X_test2)
        
        # Train on first fold, predict on second
        model.fit(X_train1, Y_train1)
        pred1 = model.predict(X_test1)
        
        # Train on second fold, predict on first
        model.fit(X_train2, Y_train2)
        pred2 = model.predict(X_test2)
        
        # For regression-based models, round and clip predictions to valid classes
        if 'Regression' in name:
            pred1 = np.rint(pred1).astype(int)
            pred1 = np.clip(pred1, 0, len(class_names) - 1)
            
            pred2 = np.rint(pred2).astype(int)
            pred2 = np.clip(pred2, 0, len(class_names) - 1)
        
        # Combine actual and predicted labels from both folds
        actual = np.concatenate([Y_test1, Y_test2])
        predicted = np.concatenate([pred1, pred2])
        
        # Compute performance metrics
        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        
        # Print results
        print(f"Results for {name}:")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.4f}")
        print("--------------------------------------------------------\n")
    
    print("All classifiers have been evaluated successfully!")


if __name__ == '__main__':
    main()
