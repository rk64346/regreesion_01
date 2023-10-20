#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Q1. Explain the difference between simple linear regression and multiple linear regression. Provide an
example of each."""


# In[2]:


"""Simple Linear Regression:

Models the relationship between one independent variable (X) and a dependent variable (Y).
Equation: 
�
=
�
0
+
�
1
∗
�
+
�
Y=b0+b1∗X+ε.
Example: Predicting salary based on years of experience.
Multiple Linear Regression:

Models the relationship between multiple independent variables (X1, X2, ..., Xn) and a dependent variable (Y).
Equation: 
�
=
�
0
+
�
1
∗
�
1
+
�
2
∗
�
2
+
.
.
.
+
�
�
∗
�
�
+
�
Y=b0+b1∗X1+b2∗X2+...+bn∗Xn+ε.
Example: Predicting salary based on years of experience and education level."""


# In[3]:


"""Q2. Discuss the assumptions of linear regression. How can you check whether these assumptions hold in
a given dataset?"""


# In[4]:


"""Linear regression makes several assumptions about the data in order for the model to be valid. Here are the key assumptions:

Linearity: The relationship between the independent variables and the dependent variable should be linear. This means that the change in the dependent variable is proportional to a change in the independent variable.

Independence of Errors: The errors (residuals) should be independent of each other. In other words, the error for one data point should not be correlated with the error for another data point.

Homoscedasticity (Constant Variance of Errors): The variance of the errors should be constant across all levels of the independent variables. This means that the spread of the residuals should be consistent.

Normality of Errors: The errors should be normally distributed. This assumption is important for conducting statistical tests and constructing confidence intervals.

No Multicollinearity: In multiple linear regression, the independent variables should not be highly correlated with each other. This can make it difficult to separate out the individual effects of the independent variables.

No Outliers: There should be no outliers in the data. Outliers can disproportionately influence the regression model."""


# In[5]:


"""Q3. How do you interpret the slope and intercept in a linear regression model? Provide an example using
a real-world scenario."""


# In[6]:


"""Intercept (
�
0
b 
0
​
 ):

The intercept represents the predicted value of the dependent variable when all independent variables are zero.
In practical terms, it's the value of the dependent variable when all predictors have no effect.
However, in many real-world scenarios, a zero value for all predictors might not be meaningful. So, the intercept's interpretation should be made in the context of your specific variables.
Slope (
�
1
b 
1
​
 ):

The slope represents the change in the dependent variable for a one-unit change in the independent variable, holding all other independent variables constant.
It indicates the strength and direction of the relationship between the predictor and the dependent variable.
For example, if 
�
1
=
0.5
b 
1
​
 =0.5 for a predictor 
�
X, it means that a one-unit increase in 
�
X is associated with a 0.5-unit increase in the dependent variable, assuming all other factors remain constant.
Example:

Let's consider a real-world scenario:

Scenario: Predicting House Prices

Dependent Variable (
�
Y): House Price (in thousands of dollars)
Independent Variable (
�
X): Square Footage of the House (in square feet)
If we have a linear regression model:

House Price
=
50
+
0.1
×
Square Footage
+
�
House Price=50+0.1×Square Footage+ε"""


# In[7]:


"""Q4. Explain the concept of gradient descent. How is it used in machine learning?"""


# In[8]:


"""Gradient Descent is an optimization algorithm used to minimize a function (usually a loss function) by iteratively moving in the direction of steepest decrease in the function. It's particularly crucial in training machine learning models.

Here's how it works in short:

Initialize Parameters: Start with initial guesses for the model's parameters.

Calculate Gradient: Compute the gradient (partial derivatives) of the loss function with respect to each parameter. The gradient points in the direction of the steepest increase in the loss.

Update Parameters: Adjust the parameters in the opposite direction of the gradient by a small step (learning rate). This "descends" the loss surface.

Repeat: Keep recalculating the gradient and updating the parameters until a stopping criterion is met (e.g., a certain number of iterations, or a sufficiently small change in loss)."""


# In[9]:


"""Q5. Describe the multiple linear regression model. How does it differ from simple linear regression?"""


# In[10]:


"""Description: Multiple linear regression is a statistical model that extends simple linear regression by considering multiple independent variables to predict a dependent variable.

Equation: 
�
=
�
0
+
�
1
∗
�
1
+
�
2
∗
�
2
+
.
.
.
+
�
�
∗
�
�
+
�
Y=b0+b1∗X1+b2∗X2+...+bn∗Xn+ε

Variables:

�
Y is the dependent variable.
�
1
,
�
2
,
.
.
.
,
�
�
X1,X2,...,Xn are the independent variables.
�
0
b0 is the intercept.
�
1
,
�
2
,
.
.
.
,
�
�
b1,b2,...,bn are the coefficients for the respective independent variables.
�
ε represents the error term."""


# In[11]:


"""Q6. Explain the concept of multicollinearity in multiple linear regression. How can you detect and
address this issue?"""


# In[12]:


"""Multicollinearity in multiple linear regression occurs when two or more independent variables are highly correlated with each other. This can cause problems in the regression model because it becomes difficult to separate out the individual effects of the correlated variables.

Detecting Multicollinearity:

Correlation Matrix: Calculate the correlation coefficients between all pairs of independent variables. If there are high correlations (close to 1 or -1), it indicates potential multicollinearity.

VIF (Variance Inflation Factor): VIF measures how much the variance of an estimated regression coefficient increases if your predictors are correlated. A high VIF (>10) suggests multicollinearity."""


# In[13]:


"""Q7. Describe the polynomial regression model. How is it different from linear regression?"""


# In[14]:


"""Polynomial Regression:

Polynomial regression is a type of regression analysis where the relationship between the independent variable (
�
X) and the dependent variable (
�
Y) is modeled as an 
�
n-th degree polynomial. This means that instead of fitting a straight line, we fit a curve to the data.

The equation for polynomial regression is:

�
=
�
0
+
�
1
∗
�
+
�
2
∗
�
2
+
.
.
.
+
�
�
∗
�
�
+
�
Y=b0+b1∗X+b2∗X 
2
 +...+bn∗X 
n
 +ε

Here, 
�
n represents the degree of the polynomial, and 
�
0
,
�
1
,
�
2
,
.
.
.
,
�
�
b0,b1,b2,...,bn are the coefficients."""


# In[15]:


"""Q8. What are the advantages and disadvantages of polynomial regression compared to linear
regression? In what situations would you prefer to use polynomial regression?"""


# In[16]:


"""Advantages of Polynomial Regression:

Captures Non-Linear Relationships
Increased Model Flexibility
Improved Fit in Non-Linear Scenarios
Disadvantages of Polynomial Regression:

Risk of Overfitting
Reduced Interpretability of Coefficients
Potential Loss of Generalization to New Data
When to Use Polynomial Regression:

Clear Curvilinear Relationships
Limited Domain of Data with Curvature
Uncertainty about True Relationship
Expected Interactions between Variables
Small Data Set with Complex Patterns (caution needed)"""


# In[ ]:




