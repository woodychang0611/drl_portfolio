import os
import gym
import pandas as pd
from pandas import Timestamp
import numpy as np
from datetime import datetime
import torch
import json
import colorsys

import common
from common.trainer import get_trainer
from common.market_env import MarketEnv
from common.finance_utility import finance_utility
from common.market_env import simple_return_reward, sharpe_ratio_reward

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
import rlkit.torch.pytorch_util as ptu


def get_unwrapped_env(env):
    return env._wrapped_env.unwrapped

def eval_policy(env, policy):
    done = False
    state = env.reset()
    rewards =[]
    weights=[]
    infos=[]
    while not done:
        action = policy.get_actions(state)
        state, reward, done, info = env.step(action)
        weights.append((get_unwrapped_env(env)).weights)
        rewards.append(reward)
        infos.append(info)
    rewards = np.array(rewards)
    return pd.DataFrame(weights),pd.DataFrame(infos)

def fix_action_policy(action):
    class dummy_policy(object):
        def __init__(self):
            self.get_actions  = lambda env: action
    return dummy_policy()



ptu.set_gpu_mode(True)

src = r'C:\Users\Woody\Documents\git repository\nccu-thesis\code\output\train_out_20210502_230851'
src=r'C:\Users\Woody\Documents\git repository\nccu-thesis\code\trained\train_out_20210502_234701'
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

trainer.policy =  torch.load(file)['evaluation/policy']
policy = MakeDeterministic(trainer.policy)




weights,infos = eval_policy(env,policy)
weights.to_csv('weights.csv')
print(infos)
import matplotlib.pyplot as plt

n = len(weights.columns)
col=map(lambda s:colorsys.hls_to_rgb(0.7, 0.8,1-0.5*float(s)/n),range(n))
#print(list(col))
#col= [colorsys.hls_to_rgb(0.5, 1, 0.5), "#e74c3c", "#34495e", "#2ecc71"]
plt.stackplot(weights.index,weights.T,colors=col)
plt.show()
#action = np.full(10,-1)
#action[5]=1
#policy = fix_action_policy(action)




