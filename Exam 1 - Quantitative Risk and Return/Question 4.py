#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Importing packages
import math
import numpy as np
import pandas as pd
from scipy.stats import norm


# In[8]:


# Setting parameters and defining ES calculation function
mu = 0
sigma = 1
c = [99.95, 99.75, 99.5, 99.25, 99, 98.5, 98, 97.5]

def ES_calc(c):
    return mu - sigma*norm.pdf(norm.ppf(1-c))/(1-c)


# In[9]:


# Calculating ES and storing in table
ES = [ES_calc(c/100) for c in c]
output_df = pd.DataFrame({
    "Confidence Level": c,
    "Expected Shortfall": ES
})
round(output_df,2)

