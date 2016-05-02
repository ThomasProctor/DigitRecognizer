
# coding: utf-8

# In[12]:

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score


# In[2]:

with open('datapath.txt') as f:
    datapath=f.readlines()[0].rstrip()


# In[3]:

train=pd.read_csv(datapath+'/train.csv')


# In[4]:

predictors=train.columns.drop('label')


# In[6]:

gbc=GradientBoostingClassifier()


# In[8]:

parameters={}


# In[13]:

parameters['n_estimators']=np.linspace(50,200,num=4,dtype=np.int)
parameters['max_features']=np.linspace(0.1,1.0,num=4)
parameters['min_samples_split']=np.linspace(2,10,num=4,dtype=np.int)
parameters['min_weight_fraction_leaf']=np.linspace(0.0,0.4,num=3)


# In[14]:

get_ipython().magic('pinfo GridSearchCV')


# In[15]:

model=GridSearchCV(gbc,parameters)


# In[17]:

model.fit(train[predictors],train['label'])


# In[ ]:



