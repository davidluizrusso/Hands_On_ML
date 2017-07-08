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
from sklearn.cross_validation import cross_val_score

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


## The ROC Curve



