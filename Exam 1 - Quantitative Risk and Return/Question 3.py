#!/usr/bin/env python
# coding: utf-8

# In[131]:


# Import pandas & yfinance
import pandas as pd
import yfinance as yf

# Import numpy
import numpy as np
from numpy import *
from numpy.linalg import multi_dot

# Import matplotlib.pyplot
import matplotlib.pyplot as plt 

# Import cufflinks
import cufflinks as cf
cf.set_config_file(offline=True, dimensions=((1000,600))) # theme= 'henanigans'

# Import plotly express for EF plot
import plotly.express as px
# px.defaults.template = "plotly_dark"
px.defaults.width, px.defaults.height = 1000, 600

# Machine info & package version
from watermark import watermark
get_ipython().run_line_magic('load_ext', 'watermark')
# %watermark -a "Kannan Singaravelu" -u -d -v -m -iv

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 4)


# In[132]:


# Create a user defined function
def binomial_option(spot, strike, rate, sigma, time, steps, output=2):
    """
    binomial_option(spot, strike, rate, sigma, time, steps, output=0)
    
    Function to calculate binomial option pricing for european call option
    
    Params
    ------
    spot       -int or float    - spot price
    strike     -int or float    - strike price
    rate       -float           - interest rate
    time       -int or float    - expiration time
    steps      -int             - number of time steps
    output     -int             - [0: price, 1: payoff, 2: option value, 3: option delta]
    
    Returns
    --------
    out: ndarray
    An array object of price, payoff, option value and delta as specified by the output flag
    
    """
    # params
    ts = time / steps
    u  = 1 + sigma*sqrt(ts) 
    v  = 1 - sigma*sqrt(ts)
    p  = 0.5 + rate *sqrt(ts) / (2*sigma)
    df = 1/(exp(-rate*ts))
    
    # initialize the arrays
    px = zeros((steps+1, steps+1))
    cp = zeros((steps+1, steps+1))
    V = zeros((steps+1, steps+1))
    d = zeros((steps+1, steps+1))
    
    # binomial loop : forward loop
    for j in range(steps+1):
        for i in range(j+1):
            px[i,j] = spot * power(v,i) * power(u,j-i)
            cp[i,j] = maximum(px[i,j] - strike, 0)
         
    # reverse loop
    for j in range(steps+1, 0, -1):
        for i in range(j):
            if (j==steps+1):
                V[i,j-1] = cp[i,j-1]
                d[i,j-1] = 0 
            else:
                V[i,j-1] = df*(p*V[i,j]+(1-p)*V[i+1,j])
                d[i,j-1] = (V[i,j]-V[i+1,j])/(px[i,j]-px[i+1,j])
    
    results = around(px,2), around(cp,2), around(V,2), around(d,4)

    return results[output]


# <h4>Option value for a range of Volatilities</h4>

# In[133]:


vols = arange(0.05, 0.81, 0.05)
prices_1 = []
for vol in vols:
    price = binomial_option(100, 100, 0.05, vol, 1, 4, 2)[0,0]
    prices_1.append(price)

results_1 = pd.DataFrame({
    "Volatility": vols,
    "Price": prices_1
})
results_1


# In[134]:


# Transposed for presentation in the report
results_1.T


# In[135]:


fig = px.scatter(
        results_1, x='Volatility', y='Price', 
        title="Binomial Option Price at Different Volatilities")

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# <h4>Option value for option vol=0.2 as time step increases</h4>

# In[136]:


steps = range(4,51)
prices_2 = []
for step in steps:
    price = binomial_option(100, 100, 0.05, 0.2, 1, step, 2)[0,0]
    prices_2.append(price)

results_2 = pd.DataFrame({
    "Step": steps,
    "Price": prices_2})


# In[137]:


# Transposed for presentation in the report
results_2.T.loc[:, '0': '18']


# In[138]:


results_2.T.loc[:, '19': '36']


# In[139]:


results_2.T.loc[:, '36': '46']


# In[140]:


fig = px.scatter(
        results_2, x='Step', y='Price', 
        title="Binomial Option Price as Time Step Increases to 50")

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# <h4>For Illustration Only: Convergence of Binomial Option Price as Time Step Increases</h4>

# In[141]:


# Increasing Time Step to 150
steps = range(4,150)
prices_2 = []
for step in steps:
    price = binomial_option(100, 100, 0.05, 0.2, 1, step, 2)[0,0]
    prices_2.append(price)

results_2 = pd.DataFrame({
    "Step": steps,
    "Price": prices_2
})

fig = px.line(
        results_2, x='Step', y='Price', 
        title="Binomial Option Price as Time Step Increases to 150")

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# In[142]:


# Increasing Time Step to 500
steps = range(4,500)
prices_2 = []
for step in steps:
    price = binomial_option(100, 100, 0.05, 0.2, 1, step, 2)[0,0]
    prices_2.append(price)

results_2 = pd.DataFrame({
    "Step": steps,
    "Price": prices_2
})

fig = px.line(
        results_2, x='Step', y='Price', 
        title="Binomial Option Price as Time Step Increases to 500")

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()

