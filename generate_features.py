import sys
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from common.finance_utility import finance_utility
from collections import OrderedDict

data_sources={
    "VIX":("yahoo", "^VIX", "raw"),
   # "Dow Jones Industrial Average":("yahoo", "^DJI", "raw"),
   # "NASDAQ Composite Index":("fred","NASDAQCOM", "raw"),
    "US Dollar/USDX":("yahoo","DX-Y.NYB", "raw"),
    "Crude Oil Prices: Brent - Europe":("fred","DCOILBRENTEU","raw"),
    "GOLD":("fred","GOLDPMGBD228NLBM","raw"),
    "5-Year Treasury Constant Maturity Rate":("fred","DGS5","rate"),
    "10-Year Treasury Constant Maturity Rate":("fred","DGS10","rate"),    
    "30-Year Treasury Constant Maturity Rate":("fred","DGS30","rate"),
    "5-Year Breakeven Inflation Rate":("fred", "T5YIE","rate"),
    "10-Year Breakeven Inflation Rate": ("fred","T10YIE","rate"),
}

use_inv_as_feature=False
if(use_inv_as_feature):
    current_folder = os.path.dirname(__file__)
    inv = pd.read_csv( os.path.join(current_folder, './data/selected_investments.csv'), index_col=0)
    for name in inv.index:
        data_sources[name] = ("yahoo", name, "raw")

sd = datetime(2007,1,1)
ed = datetime(2021,4,15)

features_dataframe= pd.DataFrame(index=pd.date_range(start=sd,end=ed, freq='D'))
features_dataframe.index.name="Date"
for name in sorted(data_sources.keys()):
    src, symbol,kind = data_sources[name]
    if (src=="yahoo"):
        series = (pdr.get_data_yahoo(symbols=symbol, start=sd, end=ed)["Adj Close"]).rename(name)
    elif (src=="fred"):
        series = (pdr.get_data_fred(symbols=symbol, start=sd, end=ed))[symbol].rename(name)

    if kind == "rate":
        features_dataframe[name]=series
    elif kind =="raw":
        for period in (5,20):            
            extended_name = f"{name}_std_{period}"
            features_dataframe[extended_name] = series.dropna().rolling(period).std()
            extended_name = f"{name}_skew_{period}"
            features_dataframe[extended_name] = series.dropna().rolling(period).skew()
            extended_name = f"{name}_kurt_{period}"
            features_dataframe[extended_name] = series.dropna().rolling(period).kurt()
    else:
        raise Exception(f"{kind} not supported")
        pass
#print(features_dataframe)
features_dataframe = features_dataframe.dropna()
features_dataframe.to_csv('./data/features_v03.csv')