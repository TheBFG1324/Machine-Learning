import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from dbn.tensorflow import SupervisedDBNClassification
from dbn import SupervisedDBNClassification

# Loading dataset
digits = load_digits()
X, Y = digits.data, digits.target

# Data scaling
X = (X / 16).astype(np.float32)

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# c. How many samples are in the test set?
print(f"Number of samples in the test set: {X_test.shape[0]}")

# d. How many samples are in the training set?
print(f"Number of samples in the training set: {X_train.shape[0]}")

# e. How many features are in the test set?
print(f"Number of features in the test set: {X_test.shape[1]}")

# f. How many features are in the training set?
print(f"Number of features in the training set: {X_train.shape[1]}")

# g. How many classes are there?
print(f"Number of classes: {len(np.unique(Y))}")

# h. List the classes.
print(f"Classes: {np.unique(Y)}")

# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=32,
                                         activation_function='relu',
                                         dropout_p=0.2)
classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore it
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)
print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))