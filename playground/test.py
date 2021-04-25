import os
import sys
import math
from pandas.io.pytables import IndexCol
current_folder = os.path.dirname(__file__)
sys.path.append(os.path.join('current_folder','./..'))

import common 

import numpy as np
import pandas as pd
from itertools import combinations
from pandas import Timestamp
from common.finance_utility import finance_utility

i = "./data/investments_returns_train.csv"
df =pd.read_csv(i, parse_dates=['Date'], index_col=['Date'])
result_df = pd.DataFrame(index=df.columns)
result_df['std'] = None
result_df['cagr'] = None

for index in result_df.index:
    prices = finance_utility.prices_from_returns(df[index])
    duration = (df[index].index[-1] -df[index].index[0]).days
    result_df.at[index,'std']=math.sqrt(252)* df[index].std()
    result_df.at[index,'mdd']=finance_utility.drawdown(prices)
    result_df.at[index,'cagr']=finance_utility.cagr(prices[0], prices[-1], duration)
#finance_utility.plot_drawdown(df['SPY'])    
print(result_df)
result_df.to_csv('train.csv')