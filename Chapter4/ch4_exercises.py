# Exercises for Chapter 4

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




