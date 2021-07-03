import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common.matplotlib_extend import set_matplotlib_style, get_graph_path
import pandas_datareader as pdr
from datetime import datetime
from common.finance_utility import finance_utility
import math

def generate_sp500_graph():
    sd = datetime(2017,3,1)
    ed = datetime(2021,3,15)
    df_sp500 = (pdr.get_data_fred(symbols='SP500', start=sd, end=ed)["SP500"]).rename('SP500')

    split_date = datetime(2019,3,1)
    df1 = df_sp500[df_sp500.index < split_date]
    df2 = df_sp500[df_sp500.index >= split_date]

    for df in (df1,df2):
        days = (df.index[-1] - df.index[0]).days
        print(finance_utility.cagr(df[0],df[-1],days))
        print(finance_utility.drawdown(df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey='row')
    axes[0].plot(df1.index,df1.values)
    axes[0].set_title(f'Period 1')
    axes[1].plot(df2.index,df2.values)
    axes[1].set_title(f'Period 2')
    for ax in axes:
        ax.set_xlabel('Date')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.suptitle('S\&P 500')
    plt.tight_layout()
    plt.savefig(get_graph_path('sp500.png'))



def generate_etf_graph():
    current_folder = os.path.dirname(__file__)
    ret_csv_train = os.path.join(current_folder, './data/investments_returns_train.csv')
    ret_csv_val = os.path.join(current_folder, './data/investments_returns_validation.csv')
    df_ret_train = pd.read_csv(ret_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_ret_val = pd.read_csv(ret_csv_val, parse_dates=['Date'], index_col=['Date'])
    def_ret = pd.concat([df_ret_train,df_ret_val])
    cagrs=[]
    stds=[]
    for etf in def_ret.columns:
        price = finance_utility.prices_from_returns(def_ret[etf])
        ret = def_ret[etf]
        days = (price.index[-1] - price.index[0]).days
        cagr =finance_utility.cagr(price[0],price[-1],days)
        std = ret.std()*math.sqrt(252)
        cagrs.append(cagr)
        stds.append(std)
    plt.ylabel('CAGR (\%)')
    plt.xlabel('Std (\%)')    
    plt.title('Performance of ETF selection')
    plt.scatter(stds,cagrs, s=30)
    plt.savefig(get_graph_path('etfs.png'))
set_matplotlib_style()
#generate_sp500_graph()
generate_etf_graph()

exit()


