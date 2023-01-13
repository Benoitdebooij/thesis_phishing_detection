#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[3]:


data_phishing = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\Code\dataset_phishing_final.csv')
data_phishing.drop(data_phishing.filter(regex="Unname"),axis=1, inplace=True)
data_phishing


# In[4]:


data_legit = pd.read_csv(r'C:\Users\benoi\Documents\Thesis_DSS\finalfinal_legit_dataset.csv')
data_legit.drop(data_legit.filter(regex="Unname"),axis=1, inplace=True)
data_legit


# In[5]:


frames = [data_phishing, data_legit]
final_dataset = pd.concat(frames)


# In[ ]:


final_dataset.to_csv("final_dataset_v3.csv")

