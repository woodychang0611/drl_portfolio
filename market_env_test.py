import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
import pandas as pd
from pandas import Timestamp
import os
import common
from common.trainer import get_sac_model
import numpy as np
gym.envs.register(id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)


current_folder = os.path.dirname(__file__)
ret_csv_train = os.path.join(current_folder, './data/investments_returns_train.csv')
ret_csv_val = os.path.join(current_folder, './data/investments_returns_validation.csv')
features_csv = os.path.join(current_folder, './data/features.csv')
df_ret_train = pd.read_csv(ret_csv_train, parse_dates=['Date'], index_col=['Date'])
df_ret_val = pd.read_csv(ret_csv_val, parse_dates=['Date'], index_col=['Date'])
df_feature = pd.read_csv(features_csv, parse_dates=['Date'], index_col=['Date'])

env_train = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_train, features=df_feature,
                                      trade_freq='weeks', show_info=True, trade_pecentage=0.2))

env_val = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_val, features=df_feature,
                                    trade_freq='weeks', show_info=True, trade_pecentage=1.0))


env = env_val
model = get_sac_model(env=env_train, hidden_sizes=[256, 256])
state = env.reset()

done = False
policy = model.policy
while not done:
    actions = policy.get_actions(state)
    state, reward, done, info = env.step(actions)
