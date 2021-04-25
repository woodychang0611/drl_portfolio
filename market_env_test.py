import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import pandas as pd
from pandas import Timestamp
import os
import common
from common.trainer import get_sac_model
from common.market_env import MarketEnv
import rlkit.torch.pytorch_util as ptu
import numpy as np
from datetime import datetime
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from common.finance_utility import finance_utility
import torch

def load_dataset():
    current_folder = os.path.dirname(__file__)
    ret_csv_train = os.path.join(current_folder, './data/investments_returns_train.csv')
    ret_csv_val = os.path.join(current_folder, './data/investments_returns_validation.csv')
    features_csv = os.path.join(current_folder, './data/features_v03.csv')
    df_ret_train = pd.read_csv(ret_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_ret_val = pd.read_csv(ret_csv_val, parse_dates=['Date'], index_col=['Date'])
    df_feature = pd.read_csv(features_csv, parse_dates=['Date'], index_col=['Date'])
    return df_ret_train, df_ret_val, df_feature


gym.envs.register(id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)

def fix_action_policy(action):
    class dummy_policy(object):
        def __init__(self):
            self.get_actions  = lambda env: action
    return dummy_policy()


def eval_policy(env, policy, df=None):
    done = False
    state = env.reset()
    while not done:
        action = policy.get_actions(state)
        state, reward, done, info = env.step(action)
        print(action)
        print(reward)
        print(info)
        if (df is not None):
            df = df.append(info, ignore_index=True)
    if (df is not None):
        return df
    
df_ret_train, df_ret_val, df_feature = load_dataset()
expl_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_train, features=df_feature,
                                    trade_freq='weeks', show_info=False, trade_pecentage=1.0))

eval_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_val, features=df_feature,
                                trade_freq='weeks', show_info=False, trade_pecentage=1.0))

#Load model from pkl file
#file = r"C:\Users\Woody\Documents\git repository\nccu-thesis\code\output\saved\itr_380.pkl"
trainer = get_sac_model(env=eval_env)
#trainer.policy.parameters =  torch.load(file)['trainer/policy']


env = expl_env

returns = env._wrapped_env.unwrapped.returns
prices = finance_utility.prices_from_returns(returns['LQD'])
print(finance_utility.drawdown(prices))
policy = trainer.policy
action = np.full(10,-1)
action[5]=1
policy = fix_action_policy(action)
df = pd.DataFrame()
df = eval_policy(env,policy,df)
print(df)
df.to_csv('test.csv')