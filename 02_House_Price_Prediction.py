#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import SGD as SGD

# Import data
original=pd.read_csv("./data/train.csv")
y_train=original["SalePrice"]
df=pd.read_csv("./data/train.csv")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth',None)
pd.set_option('display.max_rows',None)


# In[69]:


for x in df.columns[1:]: # Excluding "Id" at index 0
    print("\n"+str(x)+str(df[x].unique()))


# In[70]:


'''
Take a column
check all elements until string element is found
if string element is found, skip through all other elements in column, add column name to category types
go to next column
'''
cat_cols=[]
num_cols=[]

for x in df.columns:
    if x=='MSSubClass':
        continue
    if df[x].dtypes=='object':
        cat_cols.append(x)
    else:
        num_cols.append(x)
        
# Add MSSubClass to cat_cols (mentioned in data_description.txt)
cat_cols.append('MSSubClass')

# print(cat_cols)
# print(num_cols)


# In[71]:


df=df.fillna(0)
# Non numerical values will not be replaced, so NaN values (only remain in str cols) with "None"
df=df.fillna('None')


# In[72]:


for col in cat_cols:
    if len(df[col].unique())>5:                                              # If there are more than 5 type of categories
        df["encoded_"+col]=df.groupby(col)["SalePrice"].transform("mean")    #     Do target encoding
        weight=0.5
        ovr_avg=df["encoded_"+col].mean()
        df["encoded_"+col]=[weight*x+(1-weight)*ovr_avg for x in df["encoded_"+col]]                                                  #     TODO: Do smoothing
    else:
        df=pd.get_dummies(data=df,columns=[col],dtype=int)                   # Else do One Hot Encoding
        cat_cols.remove(col)

# Feature engineering done


# In[73]:





# In[ ]:




