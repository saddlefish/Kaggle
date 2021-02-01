#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir("C:/Users/RD45077/Documents/titanic/")
import random 

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[8]:


#loading in the data
train = pd.read_csv('train.csv') #'path to file'
test = pd.read_csv('test.csv') #'path to file'
combine = [train, test] #train + test


# In[7]:


# combine = train + test
combine


# In[9]:


# combine = [train, test]
combine


# # EDA

# In[10]:


print(train.columns)
print(test.columns)


# Now that I know what columns are in my dataset I can determine what I have in terms of numerical and categorical features. Once these features are identified I can begin my EDA because now I know which visuals might be best suited for each (i.e. visualing the number of males vs females through a bar graph instead of a scatterplot), and what features might not generate useful information through descriptive statistics, among other tasks. 

# In[11]:


train.head(5)


# In[12]:


test.head(5)


# Getting a quick look at the data is great to see if there is any missing data, whether certain columns have mixed data types within each other, or if there are errors in the data. This also allows me to confirm my initial thoughts on which columns are actually categorical versus numerical. 

# In[13]:


train.info()
test.info()


# In[16]:


#descriptive stats
train.describe()


# In[15]:


train.isnull().sum()


# In[17]:


test.isnull().sum()


# Observations
# - The train dataset has 891 samples, which is a fraction of the actual number of passengers that were on the ship
# - pclass, sex, survived, sibsp, parch are categorical variable types expressed as numerical values 
# - Both the training and testing dataset share two columns that are missing data, age and cabin, while the training also has 2 missing values 
# - There is an incredible range of prices 
# - The majority of the passengers were younger, reflected by the mean age being 29.69

# # Checking correlation between the target variable and other features

# In[18]:


age_survival_plot = sns.FacetGrid(train, col = 'Survived')
age_survival_plot.map(plt.hist, 'Age', bins = 10)


# In[21]:


#age_sex_plot = sns.FacetGrid(train, col = 'Survived')
#age_sex_plot.map(plt.bar, 'Sex')
#sns.barplot(x = train.)
#not what I want to display IGNORE
sns.barplot(x = 'Sex', y = 'Survived', data = train)


# In[24]:


age_sex_plot = sns.kdeplot(train['Age'][(train['Survived'] == 0) & (train['Age'].notnull())], color = 'Red', shade = True)
age_sex_plot = sns.kdeplot(train['Age'][(train['Survived'] == 1) & (train['Age'].notnull())], color = 'Green', shade = True)
age_sex_plot.set_xlabel('Age')
age_sex_plot.set_ylabel('Freq')
age_sex_plot = age_sex_plot.legend(["Did not survive", "Did survive"])


# In[ ]:


sns.barplot()

