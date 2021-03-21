import pandas as pd 
import numpy as np
import os
import itertools
from common.finance_utility import drawdown,cagr
from common.common_utility import timestamp,find_dict_max

current_folder = os.path.dirname(__file__)



input_csv = os.path.join(current_folder,'./data/investments_67ETF.csv')
output_folder = os.path.join(current_folder,f'./output/investments_{timestamp()}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
df = pd.read_csv(input_csv,parse_dates=['Date'])
investment_names = df.columns[1:]

investment_names = investment_names[:100]
selected_investment = set(investment_names)
corr_df = df.corr()
corr_df.to_csv(os.path.join(output_folder,'corr.csv'))

selection_count = 300
keep_investment = ('SPY','QQQ')
keep_investment = ('SPY')
while len(selected_investment) >30:
    max_corr ={}
    for investment in selected_investment:
        if (investment in keep_investment):
            continue
        #max_corr[investment] = max(corr_df[investment])
        max_corr[investment] = max(corr_df[investment],key=(lambda s:s if (s!=1) else np.nan))

    key,max_value = find_dict_max(max_corr)
    if (max_value <= 0.85):
        break
    corr_df = corr_df.drop(key,axis=0)
    corr_df = corr_df.drop(key,axis=1)
    print(f'drop {key} {max_value}')
    selected_investment.remove(key)

print(f'selected_investment:{selected_investment}')
corr_df.to_csv(os.path.join(output_folder,'selected_corr.csv'))
