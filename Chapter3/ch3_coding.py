from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist

X, y = mnist['data'], mnist['target']

X.shape

y.shape

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image,
           cmap = matplotlib.cm.binary,
           interpolation = "nearest")

plt.axis("off")


plt.show()

y[36000]


X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

import numpy as np
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Training a binary classifier
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])


# Performance Measures

## Measuring Accuracy Using Cross-Validation
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5,
               cv = 3,
               scoring = "accuracy")

from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y = None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype = bool)


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf,
               X_train,
               y_train_5,
               cv = 3,
               scoring = "accuracy")


## Confusion Matrix

# from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, 
                                X_train,
                                y_train_5,
                                cv = 3)

len(y_train_pred)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

confusion_matrix(y_train_5, y_train_5)


## Precision (accuracy of predicted yes) and Recall (Percentage of yes detected)
from sklearn.metrics import precision_score, recall_score


# ### Precision with sklearn
precision_score(y_train_5, y_train_pred)


### Precision by hand
float(4228)/(2185+4228)


### Recall by sklearn
recall_score(y_train_5, y_train_pred)

### Recall by hand
float(4228)/(1193+4228)


### F1 Scores
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)


## Precision/Recall Tradeoff
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

y_scores = cross_val_predict(sgd_clf,
                            X_train,
                            y_train_5,
                            cv = 3,
                            method = "decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

y_train_pred_90 = (y_scores > 90)


# Check predictions' precision and recall
precision_score(y_train_5, y_train_pred_90)
recall_score(y_train_5, y_train_pred_90)


# The ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

## Plot the ROC curve
def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    
plot_roc_curve(fpr, tpr)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

## Train a random forest classifier and compare its ROC and ROC AUC to the SGD Classifier 
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3,
                                    method = "predict_proba")

y_scores_forest = y_probas_forest[:, 1] # take probabilities of positive class and call those scores
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc = "lower right")

roc_auc_score(y_train_5, y_scores_forest)

# Calculate precision and recall for the random forest classifier

y_train_pred_forest = cross_val_predict(forest_clf, 
                                        X_train,
                                        y_train_5,
                                        cv = 3)

precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_train_pred_forest)

# MultiClass classification

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

## use decision_scores function
some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores.max()

### np.argmax returns the index of the largest value in the array
np.argmax(some_digit_scores)

### the classes_ method returns all possible classes (distinct values of y) in the data
sgd_clf.classes_
sgd_clf.classes_[5]

# in this case, the index of the largest value happens to correspond to the class label of 5
sgd_clf.classes_[np.argmax(some_digit_scores)]

## forcing one vs. one and one vs. all
from sklearn.multiclass import OneVsOneClassifier

### fit ovo classifier 
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])

### Train a random forest; random forests can handle multiple classes naturally and don't require 
### OVO or OVA

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])

### call predict_proba to list the probabilities that the classifier assigned to each instance of the class
forest_clf.predict_proba([some_digit])
forest_clf.classes_[np.argmax(forest_clf.predict_proba([some_digit]))]

### use cross validation to evaluate these classifiers 
cross_val_score(sgd_clf, X_train, y_train, cv = 3 , scoring = "accuracy")

#### preprocess the data with scaling to improve accuracy 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled, y_train, cv = 3, scoring = "accuracy")

# Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv = 3)
conf_mx = confusion_matrix(y_train, y_train_pred) 

plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis = 1, keepdims = True)
norm_conf_mx = conf_mx/row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)

cl_a, cl_b = 3, 5

X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

# Multilabel Classification
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
knn_clf.predict([some_digit])

# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv = 3)

# Multioutput Classification

from numpy import random as rnd

noise_train = rnd.randint(0, 100, (len(X_train), 784))
noise_test = rnd.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)


















