import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from cycler import cycler

def drawdown(x: pd.core.series.Series):
    return _drawdown(x,False)

def plot_drawdown(x: pd.core.series.Series):
    return _drawdown(x,True)

def _drawdown(x: pd.core.series.Series,display):
    e = (np.maximum.accumulate(x) - x).idxmax() # end of the period
    if (isinstance(e,float) and math.isnan(e)):
        e = x.index[0]
    s = e if (e==x.index[0]) else (x[:e]).idxmax()
    if (display):
        plt.plot(x)
        plt.plot([e, s], [x[e], x[s]], 'o', color='Red', markersize=10)
        plt.show()
        return
    else:
        return min(0,(x[e]-x[s])/(1+x[s]))
    
def cagr(start,end,len):
    return math.pow(end/start , 365/len)-1

def set_matplotlib_style():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.prop_cycle']=  cycler(color= ['#66FF99','#FFCC99','#FFFF99','#FF99FF'])
    for name in matplotlib.rcParams:
        if matplotlib.rcParams[name]=='black':
            matplotlib.rcParams[name] ='#B9CAFF'
        if matplotlib.rcParams[name]=='white':
            matplotlib.rcParams[name] ='#0C0C3A'    
