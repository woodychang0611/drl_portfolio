import pandas as pd
import numpy as np
import os
import itertools
from common.finance_utility import finance_utility
from common.common_utility import timestamp, find_dict_max
import math
current_folder = os.path.dirname(__file__)


investments_return_csv = os.path.join(
    current_folder, './data/investments_67ETF.csv')
investments_summary = os.path.join(
    current_folder, './data/investments_summary.csv')
output_folder = os.path.join(
    current_folder, f'./output/investments_{timestamp()}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
investments_return = pd.read_csv(investments_return_csv, parse_dates=[
                                 'Date'], index_col=['Date'])
investments_return = investments_return.sort_index()
investments_price = pd.DataFrame(index=investments_return.index)
investments_summary = pd.read_csv(investments_summary, index_col=0)
investment_names = investments_return.columns[:]

investments_summary['mdd'] = pd.Series(dtype='float64')
investments_summary['std'] = pd.Series(dtype='float64')
investments_summary['cagr'] = pd.Series(dtype='float64')
investments_summary['sharpe'] = pd.Series(dtype='float64')

for name in investment_names:
    prices = finance_utility.prices_from_returns(1, investments_return[name])
    std = investments_return[name].std()*math.sqrt(252)
    duration = (investments_return[name].index[-1]-investments_return[name].index[0]).days
    cagr_value = finance_utility.cagr(prices[0], prices[-1], duration)
    sharpe_ratio = cagr_value/std
    mdd=finance_utility.drawdown(prices)
    investments_summary.loc[name, 'mdd']=mdd
    investments_summary.loc[name, 'std']=std
    investments_summary.loc[name, 'cagr']=cagr_value
    investments_summary.loc[name, 'sharpe_ratio']=sharpe_ratio
# Start from all investments
selected_investment=set(investment_names)
corr_df=investments_return.corr()
corr_df.to_csv(os.path.join(output_folder, 'corr.csv'))

selection_count=300
keep_investment=('SPY', 'QQQ')
keep_investment=('SPY')
while len(selected_investment) > 10:
    max_corr={}
    for investment in selected_investment:
        if (investment in keep_investment):
            continue
        max_corr[investment]=max(corr_df[investment], key = (
            lambda s: s if (s != 1) else np.nan))

    key, max_value = find_dict_max(max_corr)
    if (max_value <= 0.99999):
        break
    corr_df = corr_df.drop(key, axis=0)
    corr_df = corr_df.drop(key, axis=1)
    print(f'drop {key} {max_value}')
    selected_investment.remove(key)

print(f'selected_investment:{selected_investment}')
print(len(corr_df.columns))
corr_df = investments_summary.join(corr_df, how='right')
corr_df.to_csv(os.path.join(output_folder, 'selected_corr.csv'))
