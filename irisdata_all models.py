#!/usr/bin/env python
# coding: utf-8

# # Iris flower Data analysis

# We first import all the required api's for our data analysis using various machine learning models

# In[2]:


import pandas
from sklearn import *
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# After importing the required libraries and importing the iris dataset using pandas, We create a list containing all the festures.
# 

# In[4]:


names=['sepal-length','sepal-width','petal-length','petal-width','class']
irisdata=pandas.read_csv(r'C:\Users\user\Downloads\iris.data',names=names)


# **dataset.shape** gives us the shape of the data set and dataset.head prints the first 5 rows of the data so that we can continue with further data cleaning if required.

# In[5]:


print(irisdata.shape)
print(irisdata.head())


# **dataset.describe** is a very good way to get the descripion of the data set as it gives information about mean, count, standard deviation, maximum and 25%, 50% and 75% distribtuion

# In[38]:


print(irisdata.describe())


# In[39]:


print(irisdata.groupby('class').size())


# #### Now we plot a box plot, histogram and scatter plot in order to understand the data

# In[20]:


irisdata.plot(kind='box',subplots='True',layout=(2,2),sharex=False,sharey=False)
plt.show()


# In[21]:


irisdata.hist()
plt.show


# In[40]:


scatter_matrix(irisdata)
plt.show


# Splitting the irisdata into train and test data

# In[9]:


A=irisdata.values
X=A[:,0:4]
Y=A[:,4]
validation_size=0.20
seed=6
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)


# In[6]:


seed=31
scoring='accuracy'


# In[7]:


models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))


# ### evaluate each model in turn

# In[11]:


results=[]
names=[]

for name, model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    kfoldcrossvalidation_results=model_selection.cross_val_score(model,X_train,Y_train, cv=kfold,scoring=scoring)
    results.append(kfoldcrossvalidation_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, kfoldcrossvalidation_results.mean(), kfoldcrossvalidation_results.std())
    print(msg)


# In[12]:


results


# We can see that all the models have similar accuracies but Linear Discriminant Analysis being the highest

# In[ ]:




