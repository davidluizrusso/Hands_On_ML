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




