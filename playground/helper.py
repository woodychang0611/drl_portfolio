import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import os

def read_investments(sd=datetime(1980,1,1),ed = datetime.today()):
    investments_dataframe= pd.DataFrame(index=pd.date_range(start=sd,end=ed, freq='D'))
    investments ={
        "Bitcoin":"BTC-USD",    
        "Tesla":"TSLA",    
        "S&P500":"VFINX",
        "Bond":"VBMFX",
        "Gold": "GC=F",
    }

    investments_data = "./../data/investments_data.csv"
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
