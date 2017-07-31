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
from math import sqrt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


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

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

# exploratory data analysis

train.columns

train.describe()
# PassengerID ranges from 1 to 891
# Survived is binary 0,1; this is what we are modeling. 
# Pclass describes socio-economic status: 1 is upper, 2 is middle, 3 is lower. Average class is 2.30
# Age ranges from 0.42 to 80; average age is 29.7
# SibSP is the number of siblings, ranges from 0 to 8
# Parch is the number of parents or children, ranges from 0 to 6
# Fare is the price the passenger paid, ranging from 0 to 512

train['Survived'].value_counts() # 342 survived, 549 died
train['Pclass'].value_counts() # 491 3, 184 2, 216 1

train.apply(lambda x: x.isnull().sum()) # 177 missing Age, 687 missing cabin, 2 missing Embarked 

train[train['Fare'] == 0].shape

# separate features and target
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked']

trainX = train[features]
trainY = train['Survived']

# replace missing ages with random number ranging from 0 to 80, with a mean of 29.7 and a std of 14
trainX['Age'] = trainX.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

# replace missing Embarked values with most common Embarked variable S
trainX['Embarked'] = trainX['Embarked'].fillna('S')


trainX["Sex"][trainX["Sex"] == "male"] = 0
trainX["Sex"][trainX["Sex"] == "female"] = 1
trainX["Embarked"][trainX["Embarked"] == "S"] = 0
trainX["Embarked"][trainX["Embarked"] == "C"] = 1
trainX["Embarked"][trainX["Embarked"] == "Q"] = 2

trainX.apply(lambda x: x.isnull().sum())

# scale features
# scaler = StandardScaler()
# trainX_scaled = scaler.fit_transform(trainX[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']])

# SGD classifier
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(trainX, trainY)

cross_val_score(sgd_clf, trainX, trainY, cv = 10, scoring = "accuracy")

# Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, trainX, trainY, cv =  10, method = "predict_proba")
y_scores_forest = y_probas_forest[:, 1]

roc_auc_score(trainY, y_scores_forest)

cross_val_score(forest_clf, trainX, trainY, cv = 10, scoring = "accuracy").mean()

forest_clf.fit(trainX, trainY) 
importance = forest_clf.feature_importances_
importance = pd.DataFrame(importance, index=trainX.columns, 
                          columns=["Importance"])

############################################# Exercise 4 #############################################
### build a spam classifier 

# 1 - Download examples of spam and ham and load into a data frame
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]    
    all_words = []       
    for mail in emails:    
        with open(mail) as m:
            for i,line in enumerate(m):
                if i == 2:  #Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
    dictionary = Counter(all_words)




