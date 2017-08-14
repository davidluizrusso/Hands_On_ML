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






