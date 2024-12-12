"""
========================================================
EECS 658 - Assignment 1
Author:    Cameron Denton
Date:      September 1st, 2024
========================================================

Brief Description:
------------------
This program checks the installed versions of Python and several key libraries 
(scipy, numpy, pandas, sklearn) and prints a "Hello World!" message to confirm 
proper installation. It then implements a simple machine learning model using 
a Gaussian Naive Bayes classifier on the Iris dataset. The model is evaluated 
using a 2-fold cross-validation approach to assess its predictive performance.

Inputs:
-------
- None for the check_install function.
- The NBClassifier function loads the Iris dataset from a CSV file named "iris.csv".

Outputs:
--------
- Prints out the versions of Python, scipy, numpy, pandas, and sklearn.
- Prints "Hello World!" to confirm proper installation of all packages.
- Prints the overall accuracy of the Naive Bayes classifier.
- Prints the confusion matrix for the classifier's predictions.
- Prints the precision, recall, and F1 score for each class of Iris flowers.

Collaborators:
--------------
- None.

Other Sources:
--------------
- Code references from David Johnson's lecture on Supervised Learning.
- Some commentary and structure suggested by ChatGPT.

========================================================
"""

# ========================================================
# Import necessary libraries for version checks and ML tasks
# ========================================================
import sys
import scipy
import numpy
import pandas
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             accuracy_score)
from sklearn.naive_bayes import GaussianNB


def check_install():
    """
    --------------------------------------------------------
    Function: check_install
    --------------------------------------------------------
    Description:
    Checks and prints the versions of Python, scipy, numpy, 
    pandas, and sklearn. Then prints a "Hello World!" message 
    to confirm that all packages are installed correctly.
    
    Inputs:
    - None
    
    Outputs:
    - Prints version information and confirmation message.
    --------------------------------------------------------
    """
    print("--------------------------------------------------------")
    print("Python: {}".format(sys.version))
    print("scipy: {}".format(scipy.__version__))
    print("numpy: {}".format(numpy.__version__))
    print("pandas: {}".format(pandas.__version__))
    print("sklearn: {}".format(sklearn.__version__))
    print("Hello World! All packages installed correctly!")
    print("--------------------------------------------------------\n")


def NBClassifier():
    """
    --------------------------------------------------------
    Function: NBClassifier
    --------------------------------------------------------
    Description:
    Loads the Iris dataset from 'iris.csv', splits it into two 
    folds for cross-validation (50-50 split), trains a Gaussian 
    Naive Bayes classifier on each fold and evaluates it on the 
    other. Calculates and prints the accuracy, confusion matrix, 
    and classification report to assess model performance.
    
    Inputs:
    - None
    
    Outputs:
    - Prints the accuracy score, confusion matrix, and 
      classification report of the classifier.
    --------------------------------------------------------
    """
    # Load the Iris dataset
    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = pandas.read_csv(url, names=names)
    
    # Separate features (X) and target labels (Y)
    data_array = dataset.values
    x_values = data_array[:, 0:4]  # First four columns are features
    y_values = data_array[:, 4]    # Fifth column is the target class

    # Perform a 2-fold split: 50% training data, 50% test data
    X_Fold1, X_Fold2, Y_Fold1, Y_Fold2 = train_test_split(
        x_values, y_values, test_size=0.50, random_state=1)
    
    # Initialize the Gaussian Naive Bayes model
    model = GaussianNB()
    
    # Train on Fold 1 and predict on Fold 2
    model.fit(X_Fold1, Y_Fold1)
    pred1 = model.predict(X_Fold2)
    
    # Train on Fold 2 and predict on Fold 1
    model.fit(X_Fold2, Y_Fold2)
    pred2 = model.predict(X_Fold1)

    # Combine predictions and actual labels from both folds
    actual = numpy.concatenate([Y_Fold2, Y_Fold1])
    predicted = numpy.concatenate([pred1, pred2])

    # Print results
    print("--------------------------------------------------------")
    print("Gaussian Naive Bayes Classifier Results")
    print("--------------------------------------------------------")
    print("Accuracy Score:", accuracy_score(actual, predicted))
    print("--------------------------------------------------------")
    print("Confusion Matrix:")
    print(confusion_matrix(actual, predicted))
    print("--------------------------------------------------------")
    print("Classification Report:")
    print(classification_report(actual, predicted))
    print("--------------------------------------------------------\n")


def main():
    """
    --------------------------------------------------------
    Function: main
    --------------------------------------------------------
    Description:
    The main function first checks the installation of packages 
    and their versions, then runs the NBClassifier to train and 
    evaluate a Gaussian Naive Bayes model on the Iris dataset.
    
    Inputs:
    - None
    
    Outputs:
    - None directly, but prints all necessary outputs.
    --------------------------------------------------------
    """
    check_install()
    NBClassifier()


# If this script is run directly, execute the main function
if __name__ == "__main__":
    main()
