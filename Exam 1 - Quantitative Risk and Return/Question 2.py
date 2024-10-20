#!/usr/bin/env python
# coding: utf-8

# <h3>Calculating Optimimal Portfolio Statistics</h3>

# In[165]:


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
import plotly.graph_objects as go
# px.defaults.template = "plotly_dark"
px.defaults.width, px.defaults.height = 1000, 600

# Machine info & package version
from watermark import watermark
get_ipython().run_line_magic('load_ext', 'watermark')
# %watermark -a "Kannan Singaravelu" -u -d -v -m -iv

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.precision', 4)


# In[166]:


# Known Parameters
assets = ['A','B','C','D']
returns = array([0.02, 0.07, 0.15, 0.20])[:, newaxis]
sigma = np.diag(array([0.05, 0.12, 0.17, 0.25]))
numofasset = len(assets)

rf = [0.005, 0.01, 0.015, 0.0175]

corr = array([[1, 0.3, 0.3, 0.3],
             [0.3, 1, 0.6, 0.6],
             [0.3, 0.6, 1, 0.6],
             [0.3, 0.6, 0.6, 1]])

# Covariance Matrix
cov = multi_dot([sigma, corr, sigma])

# Setting Matrix of Ones
one = np.ones((numofasset, 1))


# In[167]:


# Computing scalars A, B, C
A = multi_dot([one.T, linalg.inv(cov), one])
B = multi_dot([returns.T, linalg.inv(cov), one])
C = multi_dot([returns.T, linalg.inv(cov), returns])

A, B, C


# In[168]:


# Defining computation function
def tangent_opt_stats(r):
    wts = (linalg.inv(cov) @ (returns - r*one)) / (B - A*r)
    vol = sqrt((C-2*r*B+(r**2)*A)/((B-A*r)**2))
    rets = (C-B*r)/(B-A*r)
    slope = (rets-r)/vol
    return wts, vol, rets, slope


# In[169]:


wts = []; vols = []; rets = []; slope = []
for r in rf:
    wts.append(tangent_opt_stats(r)[0].flatten())
    vols.append(float(tangent_opt_stats(r)[1].flatten()))
    rets.append(float(tangent_opt_stats(r)[2].flatten()))
    slope.append(float(tangent_opt_stats(r)[3].flatten()))

output_df = pd.DataFrame({
    'Risk Free Return': rf,
    'Allocation': wts,
    'Volatility': vols,
    'Portfolio Return': rets,
    'Shape Ratio (Slope)': slope
})


# In[170]:


round(output_df,4)


# In[171]:


for r in rf:
    print(f"Risk-free rate: {r}, Optimal Allocation: {output_df[output_df['Risk Free Return']==r].Allocation.values}, Sigma: {output_df[output_df['Risk Free Return']==r].Volatility.values}")


# <h3>Plotting Efficient Frontier</h3>

# In[172]:


# Define function with risk free rate as argument
def plot_ef_with_risky_asset(r):
    numofportfolio = 100
    x = list(linspace(0, 20, numofportfolio))
    y = [float(x*output_df[output_df['Risk Free Return']==r/10000]['Shape Ratio (Slope)']) + float(output_df[output_df['Risk Free Return']==r/10000]['Shape Ratio (Slope)']) for x in x]
    temp = pd.DataFrame({
        'Risk': x,
        'Return': y
    })
    fig = px.scatter(
        temp, x='Risk', y='Return', 
        title=f"Efficient Frontier with Risky Asset at {r}bps Risk Free Rate"
         ).update_traces(mode='markers', marker=dict(symbol='cross'))

    fig.update_xaxes(showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.show()


# In[173]:


# Plotting efficient frontier
plot_ef_with_risky_asset(100)
plot_ef_with_risky_asset(175)


# <h3>Investigating the computed results</h3>

# <h4>Bringing back the simulations in Question 1</h4>

# In[174]:


# Known Parameters
assets = ['A','B','C','D']
returns = array([0.02, 0.07, 0.15, 0.20])
sigma = np.diag(array([0.05, 0.12, 0.17, 0.25]))
numofasset = len(assets)
numofportfolio = 10000

corr = array([[1, 0.3, 0.3, 0.3],
             [0.3, 1, 0.6, 0.6],
             [0.3, 0.6, 1, 0.6],
             [0.3, 0.6, 0.6, 1]])

# Covariance Matrix
cov = multi_dot([sigma, corr, sigma])


# In[175]:


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

temp = portfolio_simulation(returns)
temp = temp.loc[(temp['port_rets']<=80) & (temp['port_vols'] <=40)] 

# Note that upper limits are manually set for the X-axis and Y-axis values to avoid distortion of graph scale from large outliers


# <h4>Plotting the lines together with simulated risky asset portfolios as in Question 1</h4>

# <h5>When risk-free rate = 100 bps - all looks fine</h5>

# In[176]:


# Viewing the calculated tangency line with simulated risky asset portfolios

x = list(linspace(0, 40, 100))
y = [float(x*output_df[output_df['Risk Free Return']==100/10000]['Shape Ratio (Slope)']) + float(output_df[output_df['Risk Free Return']==100/10000]['Shape Ratio (Slope)']) for x in x]
line_temp = pd.DataFrame({
    'Risk': x,
    'Return': y
    })
fig1 = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title=f"Monte Carlo Simulated Portfolio with {numofportfolio} Observations"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))
fig2 = px.line(line_temp, x='Risk', y='Return', 
              title=f"Efficient Frontier with Risky Asset at {r}bps Risk Free Rate"
             ).update_traces(mode='markers', marker=dict(symbol='cross'))

fig3 = go.Figure(data=fig1.data + fig2.data)

fig3.add_scatter(
    mode='markers', 
    x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
    y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
    marker=dict(color='Red', size=20, symbol='star'),
    name = 'Max Sharpe'
).update(layout_showlegend=False)

fig3.update_xaxes(showspikes=True)
fig3.update_yaxes(showspikes=True)
fig3.show()


# <h5>When risk-free rate = 175 bps - What's going on?</h5>

# In[177]:


x = list(linspace(0, 40, 100))
y = [float(x*output_df[output_df['Risk Free Return']==175/10000]['Shape Ratio (Slope)']) + float(output_df[output_df['Risk Free Return']==175/10000]['Shape Ratio (Slope)']) for x in x]
line_temp = pd.DataFrame({
    'Risk': x,
    'Return': y
    })
fig1 = px.scatter(
    temp, x='port_vols', y='port_rets', color='sharpe_ratio', 
    labels={'port_vols': 'Expected Volatility', 'port_rets': 'Expected Return','sharpe_ratio': 'Sharpe Ratio'},
    title=f"Monte Carlo Simulated Portfolio with {numofportfolio} Observations"
     ).update_traces(mode='markers', marker=dict(symbol='cross'))
fig2 = px.line(line_temp, x='Risk', y='Return', 
              title=f"Efficient Frontier with Risky Asset at {r}bps Risk Free Rate"
             ).update_traces(mode='markers', marker=dict(symbol='cross'))

fig3 = go.Figure(data=fig1.data + fig2.data)

fig3.add_scatter(
    mode='markers', 
    x=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_vols']], 
    y=[temp.iloc[temp.sharpe_ratio.idxmax()]['port_rets']], 
    marker=dict(color='Red', size=20, symbol='star'),
    name = 'Max Sharpe'
).update(layout_showlegend=False)

fig3.update_xaxes(showspikes=True)
fig3.update_yaxes(showspikes=True)
fig3.show()


# <h5>Risky Asset Portfolio Return when Volatility is Lowest</h5>

# In[178]:


temp.iloc[temp.port_vols.idxmin()]['port_rets']/1000


# In[ ]:




