#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


diabetes_dataset = pd.read_csv('diabetes (1).csv')


# In[5]:


diabetes_dataset.head()


# In[7]:


diabetes_dataset.describe()


# In[9]:


diabetes_dataset['Outcome'].value_counts()


# In[12]:


diabetes_dataset.groupby('Outcome').mean()


# In[13]:


X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[14]:


print(X)
print(Y)


# data standardization
# 

# In[15]:


scaler = StandardScaler()


# In[18]:


scaler.fit(X)


# In[19]:


standardized_data = scaler.transform(X)


# In[20]:


print(standardized_data)


# In[21]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[22]:


print(X)


# In[23]:


print(Y)


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state = 2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


classifier = svm.SVC(kernel='linear')


# In[29]:


classifier.fit(X_train, Y_train)


# In[30]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[31]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[32]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[33]:


print('Accuracy score of the test data : ', test_data_accuracy)


# making a predictive system

# In[35]:


input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_np = np.asarray(input_data)

input_data_reshaped = input_data_as_np.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('the person is not diabetic')
else:
    print('the person is diabetic')


# In[ ]:




