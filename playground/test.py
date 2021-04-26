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
import matplotlib.pyplot as plt

i = "./data/investments_returns_train.csv"
df =pd.read_csv(i, parse_dates=['Date'], index_col=['Date'])
result_df = pd.DataFrame(index=df.columns)
result_df['std'] = None
result_df['cagr'] = None

def f(x):
    return 1/x
def f2(a,x):
    result = 10/ np.tanh(x/a)
    return result
fx_name = r'$f(x)=\frac{1}{x}$'

x=np.arange(0.1,0.9,0.01)
y=f(x)
plt.plot(x, y, label=fx_name)
for i in (0.05,0.2,0.5):
    y=f2(i,x)
    plt.plot(x, y, label=f"a: {i}")    

plt.legend(loc='upper left')
plt.show()