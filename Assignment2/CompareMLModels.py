"""
========================================================
EECS 658 - Assignment 2
Author:    Cameron Denton
Date:      September 16th, 2024
========================================================

Brief Description:
------------------
This program evaluates and compares the performance of several 
machine learning models on the Iris dataset. Performance is 
measured in terms of accuracy and confusion matrices for the 
following models:
- Linear Regression (LinearRegression)
- Polynomial Regression (degree 2) (LinearRegression)
- Polynomial Regression (degree 3) (LinearRegression)
- Naïve Bayes (GaussianNB)
- kNN (KNeighborsClassifier)
- LDA (LinearDiscriminantAnalysis)
- QDA (QuadraticDiscriminantAnalysis)

Inputs:
-------
- The dataset is read from a CSV file named "iris.csv".

Outputs:
--------
- Prints the accuracy and confusion matrix for each of the 
  listed models, using a 2-fold train-test split.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lectures.
- Some code and commentary suggested by ChatGPT.

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


def load_data():
    """
    --------------------------------------------------------
    Function: load_data
    --------------------------------------------------------
    Description:
    Loads the Iris dataset from 'iris.csv', extracts features 
    (X) and labels (y), and encodes the class labels into 
    numerical values.
    
    Inputs:
    - None
    
    Outputs:
    - X: Feature matrix (float)
    - y: Encoded label vector (int)
    - class_names: Original class names corresponding to 
                   the encoded labels
    --------------------------------------------------------
    """
    # Load data from CSV file
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pd.read_csv(url, names=names)
    
    # Extract features and labels
    data_array = dataset.values
    X = data_array[:, 0:4].astype(float)
    y = data_array[:, 4]

    # Encode string labels into integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder.classes_


def main():
    """
    --------------------------------------------------------
    Function: main
    --------------------------------------------------------
    Description:
    Loads the Iris dataset, performs a 2-fold split, and 
    evaluates multiple ML models on the data. For polynomial 
    regression, data is transformed using PolynomialFeatures. 
    After training on the first fold and predicting the second 
    (and vice versa), results from both folds are combined. 
    Accuracy and confusion matrices are printed for each model.
    
    Inputs:
    - None
    
    Outputs:
    - Prints model name, confusion matrix, and accuracy for 
      each model evaluated.
    --------------------------------------------------------
    """
    # Load the dataset
    X, y, class_names = load_data()
    
    # Split the dataset into two folds for cross-validation
    X_Fold1, X_Fold2, Y_Fold1, Y_Fold2 = train_test_split(X, y, test_size=0.5, random_state=1)
    
    # Models to be evaluated
    models = [
        ('Linear Regression', LinearRegression()),
        ('Polynomial Degree 2 Regression', LinearRegression()),
        ('Polynomial Degree 3 Regression', LinearRegression()),
        ('Naive Bayes', GaussianNB()),
        ('kNN', KNeighborsClassifier()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('QDA', QuadraticDiscriminantAnalysis()),
    ]
    
    print("========================================================")
    print("Model Evaluation Results")
    print("========================================================\n")
    
    # Evaluate each model
    for name, model in models:
        # Setup training/testing splits for each fold
        X_train1, Y_train1 = X_Fold1, Y_Fold1
        X_test1, Y_test1 = X_Fold2, Y_Fold2
        X_train2, Y_train2 = X_Fold2, Y_Fold2
        X_test2, Y_test2 = X_Fold1, Y_Fold1
        
        # Handle polynomial regression separately
        if 'Polynomial Degree 2' in name:
            poly = PolynomialFeatures(degree=2)
            # Fold 1
            X_train1_poly = poly.fit_transform(X_train1)
            X_test1_poly = poly.transform(X_test1)
            model.fit(X_train1_poly, Y_train1)
            pred1 = model.predict(X_test1_poly)
            
            # Fold 2
            X_train2_poly = poly.fit_transform(X_train2)
            X_test2_poly = poly.transform(X_test2)
            model.fit(X_train2_poly, Y_train2)
            pred2 = model.predict(X_test2_poly)
            
        elif 'Polynomial Degree 3' in name:
            poly = PolynomialFeatures(degree=3)
            # Fold 1
            X_train1_poly = poly.fit_transform(X_train1)
            X_test1_poly = poly.transform(X_test1)
            model.fit(X_train1_poly, Y_train1)
            pred1 = model.predict(X_test1_poly)
            
            # Fold 2
            X_train2_poly = poly.fit_transform(X_train2)
            X_test2_poly = poly.transform(X_test2)
            model.fit(X_train2_poly, Y_train2)
            pred2 = model.predict(X_test2_poly)
            
        elif 'Linear Regression' in name:
            # Simple Linear Regression (non-polynomial)
            model.fit(X_train1, Y_train1)
            pred1 = model.predict(X_test1)
            model.fit(X_train2, Y_train2)
            pred2 = model.predict(X_test2)
            
        else:
            # Non-regression models
            model.fit(X_train1, Y_train1)
            pred1 = model.predict(X_test1)
            model.fit(X_train2, Y_train2)
            pred2 = model.predict(X_test2)
        
        # For regression models, round and clip predictions to valid classes
        if 'Regression' in name:
            pred1 = np.rint(pred1).astype(int)
            pred1 = np.clip(pred1, 0, 2)
            pred2 = np.rint(pred2).astype(int)
            pred2 = np.clip(pred2, 0, 2)
        
        # Combine results from both folds
        actual = np.concatenate([Y_test1, Y_test2])
        predicted = np.concatenate([pred1, pred2])
        
        # Calculate confusion matrix and accuracy
        cm = confusion_matrix(actual, predicted)
        accuracy = accuracy_score(actual, predicted)
        
        # Print the results
        print(f"Results for {name}:")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {accuracy:.4f}")
        print("--------------------------------------------------------\n")
    
    print("All models have been evaluated successfully!")


if __name__ == '__main__':
    main()
