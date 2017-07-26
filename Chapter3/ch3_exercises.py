# load libraries
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import csv
import pandas as pd
import os


############################################# Exercise 1 #############################################

# Build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set.
# Hint: KNeighborsClassifier works well, just need to try a grid search on the weights and n_neighbors
# hyperparameters. 

# import MNIST data
mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']

# create training and test sets
X_train, X_test, y_train, y_test = X[:3000], X[60000:], y[:3000], y[60000:]
shuffle_index = np.random.permutation(1500)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# scale the test data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))


knn_clf = KNeighborsClassifier()

param_grid = [
    {'weights': ['uniform', 'distance'],
    'n_neighbors': [5, 10, 15, 20, 25]}
    ]
    
grid_search = GridSearchCV(knn_clf, param_grid, cv = 5, scoring = "accuracy")

grid_search.fit(X_train_scaled, y_train)

grid_search.best_params_
grid_search.best_estimator_

knn_best = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
           
best_preds = knn_best.predict(X_test_scaled)

conf_best = confusion_matrix(y_test, best_preds)



           

############################################# Exercise 2 #############################################

# Write a function that can shift an MNIST image one unit in any direction. 

def Shift_Image(im, x_shift, y_shift):
    return shift(im.reshape(28, 28), [x_shift, y_shift], cval = 0).reshape(784)
    
Shift_Image(X[1], 2, 1)
X[1]


# Write function to plot images 
plt.imshow(X[36000].reshape(28, 28),
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
           
# Write function to plot images 
plt.imshow(Shift_Image(X[36000], 5, 5).reshape(28, 28),
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")
           
X_expanded = [X]

for x_shift, y_shift in (-1, 0), (1, 0), (0, -1), (0, 1):
    shifted_images = np.apply_along_axis(func1d = Shift_Image,
                                         axis = 1,
                                         arr = X,
                                         x_shift = x_shift,
                                         y_shift = y_shift)
    X_expanded.append(shifted_images)

[item.shape for item in X_expanded]    
X_expanded = np.concatenate(X_expanded) # collapse list into single ndarray
X_expanded.shape

############################################# Exercise 3 #############################################

# load training and testing data for Titanic data set
INPUT_DIR = os.path.join("/Users/davidrusso/Documents/Programming/Python/Hands_On_ML/Chapter3", "Input")
OUTPUT_DIR = os.path.join("/Users/davidrusso/Documents/Programming/Python/Hands_On_ML/Chapter3", "Output")


      

############################################# Exercise 4 #############################################




############################################# Exercise 5 #############################################