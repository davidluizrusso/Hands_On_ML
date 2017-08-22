# Exercises for Chapter 4

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

### ### ### ### ###
### Question 1 ###
### ### ### ### ### 

# Question
#    What linear regression training algorithm can you 
#    use if you have a training set with millions of features? 

# Answer
#    Stochastic gradient descent or mini batch gradient descent; not the normal equations

### ### ### ### ###
### Question 2 ###
### ### ### ### ### 

# Question
#    Suppose the features in your training set have very different features.
#    Which algorithms might suffer from this and how? What can be done?

# Answer
#    Any of the gradient descent algorithms will suffer from varying scales. 
#    The features can be scaled prior to model fitting to avoid this problem.


### ### ### ### ###
### Question 3 ###
### ### ### ### ### 

# Question
#    Can gradient descent get stuck in a local minimum when training a logistic
#    regression model? 

# Answer
#    Provided the cost function is convex, no. 

### ### ### ### ###
### Question 4 ###
### ### ### ### ### 

# Question
#    Do all gradient descent algorithms lead to the same model provided you let 
#    them run long enough? 

# Answer
#    No, the stochastic gradient descent algorithm will bounce around the optimal solution
#    indefinitely. The same can be said for mini batch gradietn descent. If the learning rate 
#    is gradually decreased, all gradient descent algorithms will get close to the optimal solution.

### ### ### ### ###
### Question 5 ###
### ### ### ### ### 

# Question
#    Suppose you use batch gradient descent and you plot the validation error at every epoch.
#    If you notice that the validation error consistently goes up, what is likely going on? 
#    How can you fix this? 

# Answer
#    If the validation error is consistently going up at each epoch, the algorithm may not be 
#    converging (especially if the training error is also increasing). To fix this,
#    the learning rate can be changed. If it is not a learning rate issue, the model is likely
#    overfitting the training data and training should cease. 

### ### ### ### ###
### Question 6 ###
### ### ### ### ### 

# Question
#    Is it a good idea to stop mini batch gradient descent immediately when the validation 
#    error goes up? 

# Answer
#    Not necessarily. Mini batch gradient descent only uses a portion of the data at each
#    iteration so it may never reach the optimal solution and instead bounce around it indefinitely. 

### ### ### ### ###
### Question 7 ###
### ### ### ### ### 

# Question 
#    Which gradient descent algorithm (BGD, SGD, MBGD) will reach the vicinity of the optimal solution
#    the fastest? Which will actually converge? How can you make the others converge as well? 

# Answer
#    SGD will likely reach the vicinity of the optimal solution the fastest. BGD will actually converge.
#    To make SGD and MBGD converge, train the algorithms for a long time while gradually reducing the
#    learning rate. 


### ### ### ### ###
### Question 8 ###
### ### ### ### ### 

# Question 
#    Suppose you are using polynomial regression. You plot the learning curves and you notice that
#    there is a large gap between the training error and the validation error. What is happening? 
#    What are three ways to solve this? 

# Answer 
#    You may be overfitting the training data. 
#    1) You could reduce the polynomial degree that you are using. 
#    2) You could regularize the model via LASSO or Ridge or Elastic net. 
#    3) You can increase the size of the training set. 

### ### ### ### ###
### Question 9 ###
### ### ### ### ### 

# Question
#    Suppose you are using Ridge Regression and you notice the training error and validation error 
#    are about equal and fairly high. Does your model suffer from high bias or high variance? 
#    Should you increase the regularization hyperparameter (alpha) or reduce it? 

# Answer
#    Your model suffers from high bias. It is not capable of capturing the true relationship in its 
#    current form. You should reduce the hyperparameter alpha. 


### ### ### ### ###
### Question 10 ###
### ### ### ### ### 

# Question
#    Why would you want to use:
#    1) Ridge Regression over Linear Regression?
#    2) Lasso instead of Ridge Regression?
#    3) Elastic net instead of Lasso? 

# Answer
#    1) When you want to constrain the parameters in some way to avoid overfitting. Regularized
#       models tend to perform better than unregularized models.
#
#    2) Lasso is preferable to ridge regression if you want to perform feature selection in an 
#       automated fashion. If you suspect that only a few features are important, LASSO may be 
#       a better option than Ridge. Otherwise, Ridge is preferable. 
#
#    3) Elastic net is preferable over Lasso because Lasso tends to remove correlated features at 
#       random, which can lead to erratic behavior. Elastic net does require an extra parameter to train.


### ### ### ### ###
### Question 11 ###
### ### ### ### ### 

# Question
#    Suppose you want to classify pictures as outdoor/indoor and daytime/nighttime. 
#    Should you implement two LR classifiers or one Softmax regression classifier? 
#
# Answer
#    You should train two LR classifiers if you want the outputs to be distinctly outdoor/indoor
#    and daytime/night time. If you can live with classifiying outdoor pictures as outdoor/daytime
#    and outdoor/nighttime and indoor pictures are indoor/daytime and indoor/nighttime then you
#    could use a softmax classifier with the four aforementioned classes. 

### ### ### ### ###
### Question 12 ###
### ### ### ### ### 

# Question
#    Implement batch gradient descent with early stopping for softmax regression (without using
#    scikit-learn).

# Answer

# set random number seed for repro
np.random.seed(2042)

# load data
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]
y = iris["target"]

# add bias term to X
X_with_bias = np.c_[np.ones([len(X), 1]), X]

# create training, validation, and test sets without sklearn
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot
    
Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)

def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums    


n_inputs = X_train.shape[1] # two features + one bias term 
n_outputs = len(np.unique(y_train))  # three distinct classes to predict

eta = 0.01 # learning rate
n_iterations = 5001 
m = len(X_train) 
epsilon = 1e-7 # acceptance threshold 
best_loss = np.infty

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta) # scores
    Y_proba = softmax(logits) # 
    loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    error = Y_proba - Y_train_one_hot
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping!")
        break
    gradients = 1/m * X_train.T.dot(error)
    Theta = Theta - eta * gradients
    
    
logits = X_valid.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_valid)
accuracy_score
