import pymc3 as pm
from pymc3 import *
import theano
import theano.tensor as tt

import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import seaborn as sns
sns.set_style("whitegrid")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


price_demand = {15:20,14:18,13:35,12:50,11:65}
p0,d0 = list(price_demand.keys()), list(price_demand.values())

with pm.Model() as m:

    log_b = pm.Normal('log_b', sd = 5)                # priors
    c = pm.HalfNormal('c', sd = 5)                    # assume the elasticty to be non-negative

    log_d = log_b - c * np.log(p0)                    # demand model
    pm.Poisson('d0', np.exp(log_d), observed = d0)    # likelihood

    s = pm.sample(1000,tune=500, cores=1)     

    p = np.linspace(10, 16)   # price range
    print(p.shape)
    print(p.reshape(-1,1).shape)
    #d_means = np.exp(s.log_b - s.c * np.log(p).reshape(-1, 1))[:, :500]
    d_means = np.exp(s.log_b - s.c * np.log(p).reshape(-1, 1))[:, :500]
    print(d_means.shape)

    
    fig = plt.figure(figsize=(10, 5))
    plt.plot(p, d_means, c = 'k', alpha = 0.01)
    plt.plot(p0, d0, 'ko', markeredgewidth=1.5, markerfacecolor='w', markersize=10)
    plt.xlabel('Price ($)')
    plt.ylabel('Demand (Units)')
    plt.show() 
    

    cost = 11
    profit = (p - cost).reshape(-1,1) * d_means
    print(profit.shape)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(p,profit,c='k',alpha=0.01)

    #plt.plot(p,np.mean(profit,1).T,c='C1',lw=2,label="$\mathbb{E}[profit|P]$")
    #plt.fill_between(p,(np.mean(profit,1)-np.std(profit,1)).T,(np.mean(profit,1)+np.std(profit,1)).T,alpha=0.1,color='C1')
    #plt.plot(p,(np.mean(profit,1)+np.std(profit,1)).T,c='C1',lw=1,label="$\mathbb{E}[profit|P]\ \pm$1 sd")
    #plt.plot(p,(np.mean(profit,1)-np.std(profit,1)).T,c='C1',lw=1)
    pmax = p[np.argmax(np.mean(profit,1))]
    #plt.vlines(pmax,300,900,colors='C0',linestyles='dashed',label="argmax$_P\ \mathbb{E}[profit|P]$")

    plt.ylim(300,900)
    plt.xlabel("Price $P$")
    plt.ylabel("Profit $")

    plt.legend()
    plt.show() 






