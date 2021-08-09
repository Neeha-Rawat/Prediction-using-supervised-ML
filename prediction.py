#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics Intern at SPARKS Foundation

# # Author : Neeha Rawat

# ## To predict the percentage of an student based on the no. of study hours.

# In[1]:


import pandas as pd #library to analyse and manipulate data
import numpy as np #library to perform mathematical operations
import matplotlib.pyplot as plt #library to visualize data


# In[2]:


url="http://bit.ly/w-data"
d1=pd.read_csv(url) #to load data
print("Data imported successfully")


# In[3]:


d1.head(10) #shows first 10 rows of the dataset


# In[4]:


d1.describe()


# In[5]:


d1.info()


# ### Visualizing our dataset

# In[6]:


d1.plot(x="Hours",y="Scores",style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.show()


# In[7]:


X=d1.iloc[:,:-1].values 
y=d1.iloc[:,1].values


# ## Splitting the Data

# In[8]:


from sklearn.model_selection import train_test_split #library to split data into training and test sets
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.2,random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train,y_train)
print("Training Complete")


# In[10]:


line=regressor.coef_ * X + regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# ## Predicting the Percentage of Marks

# In[11]:


print(X_test)
y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# ##  What will be predicted score if a student studies for 9.25 hrs/ day?

# In[12]:


hours=9.25
own_pred = regressor.predict([[hours],])
print("No. of hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### If a student studies for 9.25 hrs/day, his predicted score is 93.69%

# ## Evaluating Linear Regression using Mean Absolute Error and Mean Squared Error

# In[13]:


from sklearn import metrics
print ('Mean Absolute Error :',metrics.mean_absolute_error(y_test,y_pred))
print ('Mean Squared Error :',metrics.mean_squared_error(y_test,y_pred))


# In[14]:


accuracy=regressor.score(X_test,y_test)
print("Accuracy",accuracy*100,"%")


# In[ ]:




