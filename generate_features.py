import sys
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os
import matplotlib.pyplot as plt
import math
import numpy as np
from common.finance_utility import finance_utility


data_sources={
    "VIX":("yahoo", "^VIX", "raw"),
    "VFINX":("yahoo","VFINX", "raw"),
    "QQQ":("yahoo","QQQ", "raw"),
    #"M1V":("fred","M1V","rate"),
    #"M2V":("fred","M2V","rate"),
    "Crude Oil Prices: Brent - Europe":("fred","DCOILBRENTEU","raw"),
    "5-Year Treasury Constant Maturity Rate":("fred","DGS5","rate"),
    "10-Year Treasury Constant Maturity Rate":("fred","DGS10","rate"),    
    "30-Year Treasury Constant Maturity Rate":("fred","DGS30","rate"),
    "5-Year Breakeven Inflation Rate":("fred", "T5YIE","rate"),
    "10-Year Breakeven Inflation Rate": ("fred","T10YIE","rate"),
    #"Unemployment Rate" : ("fred","UNRATE","rate"),
    "GOLD":("fred","GOLDPMGBD228NLBM","rate"),
    #"Stress Index":("fred","STLFSI2","rate"),
    
}

data_sources={
    "VIX":("yahoo", "^VIX", "raw"),
    "VFINX":("yahoo","VFINX", "raw"),
    "QQQ":("yahoo","QQQ", "raw"),
}

sd = datetime(2007,1,1)
ed = datetime(2021,4,15)

features_dataframe= pd.DataFrame(index=pd.date_range(start=sd,end=ed, freq='D'))
features_dataframe.index.name="Date"
for name in data_sources.keys():
    src, symbol,kind = data_sources[name]
    if (src=="yahoo"):
        df = (pdr.get_data_yahoo(symbols=symbol, start=sd, end=ed)["Adj Close"]).rename(name)
    elif (src=="fred"):
        df = (pdr.get_data_fred(symbols=symbol, start=sd, end=ed))[symbol].rename(name)
    features_dataframe[name]=df

#print(features_dataframe)
features_dataframe = features_dataframe.dropna()
features_dataframe.to_csv('./data/features_v02.csv')

