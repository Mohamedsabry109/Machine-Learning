
# coding: utf-8

# <h1 style="color:black" align='center'>Data preprocessing</h1>

# In[1]:


# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# # Importing our Data using pandas

# In[3]:


Dataset = pd.read_csv('unprocessed_data.csv')
Dataset


# # seperating the data to inputs and output

# In[4]:


#seperating the features "inputs"
features_matrix = Dataset.iloc[:,:-1].values #passing the data to a numpy array called features_matrix
features_matrix
print(type(features_matrix))


# In[5]:


#seperating the output -> "purchased or not"
goal_vector = Dataset.iloc[:,-1].values
goal_vector


# # Handling the missing data by replacing it with the mean

# In[6]:


# import the Imputer module which can handle the missing data
from sklearn.preprocessing import Imputer
# creating a new object
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0) #we want to replace a row -> axis=0 , with it's mean -> strategy='mean'
features_matrix[:, 1:3] = imputer.fit_transform(features_matrix[:, 1:3])#apply changes to all rows and cols from 1 to 3 
features_matrix


# ## after all in training a model we must use number so transforming a categorical data to numbers is an important step

# In[7]:


from sklearn.preprocessing import LabelEncoder #importing LabelEncoder class 
encoder = LabelEncoder() #creating a new object
features_matrix[:, 0] = encoder.fit_transform(features_matrix[:, 0]) #transforming the categorical data to numbers
features_matrix


# # don't forget to transform the output if it's a boolean one

# In[8]:


goal_vector = encoder.fit_transform(goal_vector)
goal_vector


# ## in Dealing with numbers "2" is greater than "0" which might affect the model as it will be more biased towards "2"

# In[9]:


# import the oneHotEncoder class
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0]) #creating a new object and sending the targeted data to it
features_matrix = oneHotEncoder.fit_transform(features_matrix).toarray()#transforming numbers
features_matrix


# # think about training The model 
# 
# ## we must have a training set and a test set 

# In[10]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features_matrix, goal_vector, train_size = 0.8, random_state = 0)
print(len(x_train))
print(len(x_test))
print(len(features_matrix))
# see the size percentage !
# note :
## the random state just to have the same result each time you run
## the train size is changable and it depend on several things[the data size, the problem itself, ...]


# # a large diversity in numbers leads to slowing down the computation 
# ## it's better to scale your Data 

# In[11]:


# import the library we need from our beloved sklearn !
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# take a look how the values was before and after the scalling 
print('before scalling, max is %d and min is %d'%(np.max(x_train), np.min(x_train)))
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print('after scalling, max is %d and min is %d'%(np.max(x_train), np.min(x_train)))
x_train


# In[12]:


from sklearn import preprocessing
X_scaled = preprocessing.scale(x_train)
X_scaled
X_scaled.mean(axis=0) # mean is zero
X_scaled.std(axis=0) # variance is 1


# # Scaling the data to a specific range

# In[13]:


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)) #setting the range
X_train_minmax = min_max_scaler.fit_transform(x_train)
X_train_minmax


# # Binarization

# In[14]:


binarizer = preprocessing.Binarizer().fit(X_train_minmax)  # fit does nothing
binarizer.transform(X_train_minmax)

