#!/usr/bin/env python
# coding: utf-8

# # Linear Regression on Boston Housing Dataset
# 
# This Housing dataset contains information about different houses in Boston. This data was originally a part of UCI Machine Learning Repository and has been removed now.
# 
# There are 506 samples and 13 feature variables in this dataset. The objective is to predict the value of prices of the house using the given features. 
# The dataset itself is available at https://archive.ics.uci.edu/ml/datasets/Housing . However, because we are going to use scikit-learn, we can import it right away from the scikit-learn itself.
# 
# First, we will import the required libraries.

# In[1]:


import numpy as np
import pandas as pd

#Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt

#To plot the graph embedded in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#imports from sklearn library

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# In[3]:


#loading the dataset direclty from sklearn
boston = datasets.load_boston()


# sklearn returns Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the regression targets, ‘DESCR’, the full description of the dataset, and ‘filename’, the physical location of boston csv dataset. This we can from the following Operations 

# In[4]:


print(type(boston))
print('\n')
print(boston.keys())
print('\n')
print(boston.data.shape)
print('\n')
print(boston.feature_names)


# In[5]:


print(boston.DESCR)


# In[6]:


bos = pd.DataFrame(boston.data, columns = boston.feature_names)

print(bos.head())


# In[7]:


print(boston.target.shape)


# In[8]:


bos['PRICE'] = boston.target
print(bos.head())


# ## Data preprocessing
# After loading the data, it’s a good practice to see if there are any missing values in the data. We count the number of missing values for each feature using isnull()

# In[9]:


bos.isnull().sum()


# In[10]:


print(bos.describe())


# ## Exploratory Data Analysis
# 
# Exploratory Data Analysis is a very important step before training the model. Here, we will use visualizations to understand the relationship of the target variable with other features.
# 
# Let’s first plot the distribution of the target variable $ Price $. We will use the distplot function from the seaborn library.

# In[11]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.hist(bos['PRICE'], bins=30)
plt.xlabel("House prices in $1000")
plt.show()


# We can see from the plot that the values of **PRICE** are distributed normally with few outliers. Most of the house are around 20-24 range (in $1000 scale)
# 
# Now, we create a correlation matrix that measures the linear relationships between the variables. The correlation matrix can be formed by using the corr function from the pandas dataframe library. We will use the heatmap function from the seaborn library to plot the correlation matrix.

# In[12]:


#Created a dataframe without the price col, since we need to see the correlation between the variables
bos_1 = pd.DataFrame(boston.data, columns = boston.feature_names)

correlation_matrix = bos_1.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)


# The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation.
# 
# ### Notice
# 
# 1. By looking at the correlation matrix we can see that *RM* has a strong positive correlation with *PRICE* **(0.7)** where as *LSTAT* has a high negative correlation with *PRICE* **(-0.74)**.
# 2. An important point in selecting features for a linear regression model is to check for multicolinearity. The features *RAD*, *TAX* have a correlation of **0.91**. These feature pairs are strongly correlated to each other. This can affect the model. Same goes for the features *DIS* and *AGE* which have a correlation of *-0.75*.
# 
# But for now we will keep all the features.

# In[13]:


plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = bos['PRICE']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = bos[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')


# ### Notice
# 1. The prices increase as the value of RM increases linearly. There are few outliers and the data seems to be capped at 50.
# 2. The prices tend to decrease with an increase in LSTAT. Though it doesn’t look to be following exactly a linear line.
# 
# Since it is really hard to visualise with the multiple features, we will 1st predict the house price with just one vaiable and then move to the regression with all features.
# 
# Since you saw that **'RM'** shows positive correlation with the **House Prices** we will use this variable

# In[14]:


X_rooms = bos.RM
y_price = bos.PRICE


X_rooms = np.array(X_rooms).reshape(-1,1)
y_price = np.array(y_price).reshape(-1,1)

print(X_rooms.shape)
print(y_price.shape)


# ### Splitting the data into training and testing sets
# 
# SInce we need to test our model, we split the data into training and testing sets. We train the model with 80% of the samples and test with the remaining 20%. We do this to assess the model’s performance on unseen data. 
# 
# To split the data we use train_test_split function provided by scikit-learn library. We finally print the shapes of our *training* and *test* set to verify if the splitting has occurred properly.

# In[15]:


X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_rooms, y_price, test_size = 0.2, random_state=5)


# In[16]:


print(X_train_1.shape)
print(X_test_1.shape)
print(Y_train_1.shape)
print(Y_test_1.shape)


# ### Training and testing the model
# Here we use scikit-learn’s LinearRegression to train our model on both the training and check it on the test sets

# In[17]:


reg_1 = LinearRegression()
reg_1.fit(X_train_1, Y_train_1)


# In[18]:


# model evaluation for training set

y_train_predict_1 = reg_1.predict(X_train_1)
rmse = (np.sqrt(mean_squared_error(Y_train_1, y_train_predict_1)))
r2 = round(reg_1.score(X_train_1, Y_train_1),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[19]:


# model evaluation for test set

y_pred_1 = reg_1.predict(X_test_1)
rmse = (np.sqrt(mean_squared_error(Y_test_1, y_pred_1)))
r2 = round(reg_1.score(X_test_1, Y_test_1),2)

print("The model performance for training set")
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")


# In[20]:


prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1) 
plt.scatter(X_rooms,y_price)
plt.plot(prediction_space, reg_1.predict(prediction_space), color = 'black', linewidth = 3)
plt.ylabel('value of house/1000($)')
plt.xlabel('number of rooms')
plt.show()


# ### Regression Model for All the variables
# 
# Now we will create a model considering all the features in the dataset. The process is almost the same and also the evaluation model but in this case the visualization will not be possible in a 2D space.

# In[21]:


X = bos.drop('PRICE', axis = 1)
y = bos['PRICE']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[23]:


reg_all = LinearRegression()
reg_all.fit(X_train, y_train)


# In[24]:


# model evaluation for training set

y_train_predict = reg_all.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = round(reg_all.score(X_train, y_train),2)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


# In[25]:


# model evaluation for test set

y_pred = reg_all.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
r2 = round(reg_all.score(X_test, y_test),2)

print("The model performance for training set")
print("--------------------------------------")
print("Root Mean Squared Error: {}".format(rmse))
print("R^2: {}".format(r2))
print("\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[26]:


get_ipython().run_line_magic('timeit', 'cv_results = cross_val_score(reg_all, X, y, cv = 5)')


# In[27]:


print(cv_results)
round(np.mean(cv_results),2)


# In[ ]:


get_ipython().run_line_magic('timeit', 'cvresults_3 = cross_val_score(reg_all, X, y, cv = 3)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'cv_results_10 = cross_val_score(reg_all, X, y, cv = 10)')


# In[ ]:


from sklearn.linear_model import Ridge, Lasso


# In[ ]:


ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)


# In[ ]:


lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)


# In[ ]:


names = boston.feature_names
lasso_coef = lasso.fit(X_train,y_train).coef_


# In[ ]:


plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names, rotation = 60)
plt.ylabel('Coefficient')
plt.show()


# In[ ]:


alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
ridge = Ridge(normalize=True)
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge_cv_scores = cross_val_score(ridge,X_train,y_train,cv = 10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))


# In[ ]:


def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# In[ ]:


display_plot(ridge_scores, ridge_scores_std)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




