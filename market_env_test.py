import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
import pandas as pd
from pandas import Timestamp
import os
gym.envs.register(
    id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)



NormalizedBoxEnv(gym.make('MarketEnv-v0',investments_returns=1,features=2))
current_folder = os.path.dirname(__file__)
f = os.path.join(current_folder,'./data/selected_investments.csv')
df = pd.read_csv(f, parse_dates=[
    'Date'], index_col=['Date'])
t = Timestamp('2016-03-03 00:00:00')
print(df[df.index > t])
