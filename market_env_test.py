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
from common.trainer import get_trainer
from common.market_env import MarketEnv
import rlkit.torch.pytorch_util as ptu
import numpy as np
from datetime import datetime
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from common.finance_utility import finance_utility
import torch
import json
from common.market_env import simple_return_reward, sharpe_ratio_reward

def eval_policy(env, policy, df=None):
    done = False
    state = env.reset()
    rewards =[]
    while not done:
        action = policy.get_actions(state)
        state, reward, done, info = env.step(action)

        #print(f'state: {state}')    
        #print(f'action: {action}')
        #print(f'reward: {reward}')
        rewards.append(reward)
        print(info)
        if (df is not None):
            df = df.append(info, ignore_index=True)
    rewards = np.array(rewards)
    print(f'mean:{np.mean(rewards)}')
    print(f'std:{np.std(rewards)}')
    print(f'ratio:{np.mean(rewards)/np.std(rewards)}')
    if (df is not None):
        return df

def fix_action_policy(action):
    class dummy_policy(object):
        def __init__(self):
            self.get_actions  = lambda env: action
    return dummy_policy()


def get_unwrapped_env(env):
    return env._wrapped_env.unwrapped


src = r'C:\Users\Woody\Documents\git repository\nccu-thesis\code\output\train_out_20210502_220229'
with open(os.path.join(src,'variant.json')) as json_file:
    variant=json.load(json_file)

expl_env_kwargs = variant['expl_env_kwargs']
eval_env_kwargs = variant['eval_env_kwargs']    
trainer_kwargs = variant['trainer_kwargs']
df_ret_train = pd.read_csv(os.path.join(src,'df_ret_train.csv'), parse_dates=['Date'], index_col=['Date'])
df_ret_val = pd.read_csv(os.path.join(src,'df_ret_val.csv'), parse_dates=['Date'], index_col=['Date'])
df_feature = pd.read_csv(os.path.join(src,'df_feature.csv'), parse_dates=['Date'], index_col=['Date'])
gym.envs.register(id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)


#todo should parse it from json file
reward_func = simple_return_reward
expl_env_kwargs['reward_func'] = simple_return_reward
eval_env_kwargs['reward_func'] = simple_return_reward

expl_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_train, features=df_feature,
                                        **expl_env_kwargs))

eval_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', returns=df_ret_val, features=df_feature,
                                        **eval_env_kwargs))

env = eval_env
trainer = get_trainer(env=env,**trainer_kwargs)
file =os.path.join(src,'params.pkl')
v = torch.load(file)
print(type(trainer.policy))
print(v['evaluation/policy'])

trainer.policy.parameters =  torch.load(file)['evaluation/policy']
policy = trainer.policy
df = pd.DataFrame()
df = eval_policy(env,policy,df)
exit()






trainer = get_sac_model(env=eval_env)
#trainer.policy.parameters =  torch.load(file)['trainer/policy']


env = expl_env
unwrapped_env = get_unwrapped_env(env)
returns = env._wrapped_env.unwrapped.returns
features = env._wrapped_env.unwrapped.features
#features.to_csv('features.csv')
#exit()

prices = finance_utility.prices_from_returns(returns['LQD'])
print(finance_utility.drawdown(prices))
policy = trainer.policy
action = np.full(10,-1)
action[5]=1
policy = fix_action_policy(action)
df = pd.DataFrame()
df = eval_policy(env,policy,df)
df.to_csv('test.csv')