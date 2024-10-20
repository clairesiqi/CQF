#!/usr/bin/env python
# coding: utf-8

# In[98]:


# Import pandas & yfinance
import pandas as pd
import yfinance as yf

# Import numpy
import numpy as np
from numpy import *
from numpy.linalg import multi_dot

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


# In[99]:


# Known Parameters
assets = ['A','B','C','D']
returns = array([0.02, 0.07, 0.15, 0.20])
sigma = np.diag(array([0.05, 0.12, 0.17, 0.25]))
numofasset = len(assets)
numofportfolio = 700

corr = array([[1, 0.3, 0.3, 0.3],
             [0.3, 1, 0.6, 0.6],
             [0.3, 0.6, 1, 0.6],
             [0.3, 0.6, 0.6, 1]])

# Covariance Matrix
cov = multi_dot([sigma, corr, sigma])
cov


# In[100]:


# Simulating Portfolio
def portfolio_simulation(returns):
    rets=[]; vols=[]; wts=[]
    for i in range(numofportfolio):
        weights = array([random.uniform(-1, 1) for i in range(0,4)])[:, newaxis]
         
        # weights = random.random(numofasset)[:, newaxis]
        weights /= sum(weights)
        
        rets.append(weights.T @ returns[:,newaxis])
        vols.append(sqrt(multi_dot([weights.T, cov, weights])))
        
        wts.append(weights.flatten())
    portdf = 100*pd.DataFrame({
        'port_rets': array(rets).flatten(),
        'port_vols': array(vols).flatten(),
        'weights': list(array(wts))
    })
    portdf['sharpe_ratio'] = portdf['port_rets'] / portdf['port_vols']
        
    return round(portdf, 2)


# In[101]:


temp = portfolio_simulation(returns)
temp = temp.loc[(temp['port_rets']<=80) & (temp['port_vols'] <=40)] 

# Note that upper limits are manually set for the X-axis and Y-axis values to avoid distortion of graph scale from large outliers


# In[102]:


fig = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title=f"Monte Carlo Simulated Portfolio with {numofportfolio} Observations"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))

# Plot max sharpe 
# fig.add_scatter(
#     mode='markers', 
#     x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
#     y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
#     marker=dict(color='RoyalBlue', size=20, symbol='star'),
#     name = 'Max Sharpe'
# ).update(layout_showlegend=False)

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# <h4>Illustration Only: Hyperbolic Plane with 10000 simulations</h4>

# In[103]:


# Increasing number of random generations to 10000
numofportfolio=10000

# Cropping the plot to exclude outliers
temp = portfolio_simulation(returns)
temp = temp.loc[(temp['port_rets']<=80) & (temp['port_vols'] <=40)] 

# Plotting 
fig = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title=f"Monte Carlo Simulated Portfolio with {numofportfolio} Observations"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))

# Plot max sharpe 
# fig.add_scatter(
#     mode='markers', 
#     x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
#     y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
#     marker=dict(color='RoyalBlue', size=20, symbol='star'),
#     name = 'Max Sharpe'
# ).update(layout_showlegend=False)

fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
fig.show()


# In[ ]:




