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
    if (not isinstance(x,(pd.core.series.Series))):
        raise TypeError(f'type {type(x)} not supported')
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

def read_investments(sd=datetime(1980,1,1),ed = datetime.today()):
    investments_dataframe= pd.DataFrame(index=pd.date_range(start=sd,end=ed, freq='D'))
    investments ={
        "Bitcoin":"BTC-USD",    
        "Tesla":"TSLA",    
        "S&P500":"VFINX",
        "Bond":"VBMFX",
        "Gold": "GC=F",
    }

    investments_data = "./data/investments_data.csv"
    s=[]
    if (os.path.exists(investments_data)):
        print("Load existing data")
        investments_dataframe = pd.read_csv(investments_data, parse_dates=True,index_col='Date')
    else:
        for name in investments.keys():
            s = (pdr.get_data_yahoo(symbols=investments[name], start=sd, end=ed)["Adj Close"]).rename(name)
            #drop duplicated index
            s = s.groupby(level=0).last()
            investments_dataframe[name]=s
        investments_dataframe.to_csv(path_or_buf=investments_data,index_label='Date')
    return investments_dataframe