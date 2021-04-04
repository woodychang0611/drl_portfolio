# Generate investments for training and validation based on given criteria

import pandas as pd
import numpy as np
import os
import itertools
from common.finance_utility import finance_utility
from common.common_utility import timestamp
import math
from pandas import Timestamp

investment_count = 10
validation_start_date = Timestamp('2017-03-01')


current_folder = os.path.dirname(__file__)
investments_returns_csv = os.path.join(
    current_folder, './data/src/investments_67ETF.csv')
investments_summary = os.path.join(
    current_folder, './data/src/investments_summary.csv')
output_folder = os.path.join(
    current_folder, f'./output/investments_{timestamp()}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
investments_returns = pd.read_csv(investments_returns_csv, parse_dates=[
    'Date'], index_col=['Date'])
investments_returns = investments_returns.sort_index()
investments_prices = pd.DataFrame(index=investments_returns.index)
investments_summary = pd.read_csv(investments_summary, index_col=0)
investment_names = investments_returns.columns[:]


def average_return(s):
    return math.pow(np.prod(s+1), 1.0/len(s))-1


for name in investment_names:
    investments_prices[name] = finance_utility.prices_from_returns(
        1, investments_returns[name])


def get_aum(s):
    return investments_summary['AUM'][s]


cov_df = investments_returns.cov()
cov_df.to_csv(os.path.join(output_folder, 'cov.csv'))

for name in cov_df.columns:
    cov_df[name][name] = np.nan
cov_df = cov_df.applymap(lambda s: s if (s != 1) else np.nan)

investments_summary['mdd'] = pd.Series(dtype='float64')
investments_summary['std'] = pd.Series(dtype='float64')
investments_summary['cagr'] = pd.Series(dtype='float64')
investments_summary['sharpe'] = pd.Series(dtype='float64')

for name in investment_names:
    prices = finance_utility.prices_from_returns(1, investments_returns[name])
    # Wiener process
    std = investments_returns[name].std()*math.sqrt(252)
    duration = (investments_returns[name].index[-1] -
                investments_returns[name].index[0]).days
    cagr_value = finance_utility.cagr(prices[0], prices[-1], duration)
    sharpe = cagr_value/std
    mdd = finance_utility.drawdown(prices)
    investments_summary.loc[name, 'mdd'] = mdd
    investments_summary.loc[name, 'std'] = std
    investments_summary.loc[name, 'cagr'] = cagr_value
    investments_summary.loc[name, 'sharpe'] = sharpe
# Start from all investments
selected_investments = set(investment_names)


while len(selected_investments) > investment_count:
    # Keep the one with highest AUM (Asset Under Management)

    # compare highist
    def compare_func(k):
        return ((cov_df[k].max(), get_aum(k)))
    drop_name = sorted(selected_investments, key=compare_func)[-2]
    keep_name = sorted(selected_investments, key=compare_func)[-1]
    print(
        f'drop {drop_name} {compare_func(drop_name)}, keep {keep_name} {compare_func(keep_name)}')
    cov_df = cov_df.drop(drop_name, axis=0)
    cov_df = cov_df.drop(drop_name, axis=1)
    selected_investments.remove(drop_name)

print(f'selected_investments:{selected_investments}')
print(len(cov_df.columns))
cov_df = investments_summary.join(cov_df, how='right')
cov_df.to_csv(os.path.join(output_folder, 'selected_cov.csv'))
print(selected_investments)

#Create data for traininng 
df_train = investments_returns[investments_returns.index < validation_start_date][selected_investments]
df_train.to_csv(os.path.join(current_folder, './data/investments_train.csv'))
#Create data for Validation
df_validation = investments_returns[investments_returns.index >= validation_start_date][selected_investments]
df_validation.to_csv(os.path.join(current_folder, './data/investments_validation.csv'))
