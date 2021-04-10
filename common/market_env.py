# -*- coding: utf-8 -*-
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


def proration_weights(action):
    if action.sum() == 0:
        action = np.random.rand(*action.shape)
    return action / action.sum()


def simple_return_reward(env):
    reward = env.wealth-env.previous_wealth
    return reward


def resample_relative_changes(df, rule):
    return df.apply(lambda x: 1+x).resample(rule).prod().apply(lambda x: x-1)


class MarketEnv(gym.Env):
    def __init__(self, returns: DataFrame, features: DataFrame, show_info=False, trade_freq='days',
                 action_to_weights_func=proration_weights,
                 reward_func=simple_return_reward,
                 trade_pecentage=0.1):
        self._load_data(returns=returns, features=features, show_info=show_info, trade_freq=trade_freq)
        self._init_action_space()
        self._init_observation_space()
        self.trade_pecentage = trade_pecentage
        self.start_index, self.current_index, self.end_index = 0, 0, 0
        self.action_to_weights_func = action_to_weights_func
        self.reward_func = reward_func
        self.seed()
        self.reset()

    def _load_data(self, returns: DataFrame, features: DataFrame, show_info, trade_freq):
        resample_rules = {
            'days': 'D',
            'weeks': 'W',
            'months': 'M'
        }

        if(trade_freq not in resample_rules.keys()):
            raise ValueError(f"trade_freq '{trade_freq}' not supported, must be one of {list(resample_rules.keys())}")

        if returns.isnull().values.any():
            raise ValueError('At least one null value in investments')
        if features.isnull().values.any():
            raise ValueError('At least one null value in investments')
        # make sure dataframes are sorted
        returns = returns.sort_index().sort_index(axis=1)
        features = features.sort_index().sort_index(axis=1)

        # Scale features to -1 and 1
        for col in features.columns:
            features[col] = features[col]/max(abs(features[col].max()), abs(features[col].min()))

        # Only keep feature data within peroid of investments returns
        features = features[(features.index.isin(returns.index))]

        # resample based on trade frequency e.g. weeks or months
        returns = resample_relative_changes(returns, resample_rules[trade_freq])
        features = resample_relative_changes(features, resample_rules[trade_freq])

        self.features = features
        self.returns = returns
        if show_info:
            sd = returns.index[0]
            ed = returns.index[-1]
            print(f'Trading Frequency: {trade_freq}')
            print(f'{self.investments_count} investments loaded')
            print(f'{self.features_count} features loaded')
            print(f'Starts from {sd} to {ed}')

    def _init_action_space(self):
        action_space_size = self.investments_count
        self.min_action = np.zeros(action_space_size)
        self.max_action = np.ones(action_space_size)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

    def _init_observation_space(self):
        observation_space_size = self.investments_count+self.features_count
        self.low_state = np.full(observation_space_size, -1)
        self.high_state = np.ones(observation_space_size)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    @property
    def investments_count(self):
        return len(self.returns.columns)

    @property
    def features_count(self):
        return len(self.features.columns)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print(f'step {self.returns.index[self.current_index]}')
        if self.current_index > self.end_index:
            raise Exception(f'current_index {current_index} exceed end_index{self.end_index}')

        done = True if (self.current_index >= self.end_index) else False
        # update weight
        self.weights = self.action_to_weights_func(action)

        self.current_index += 1
        # update investments and wealth
        previous_investments = self.investments
        target_investments = self.wealth * self.weights

        # todo add trading cost
        self.investments = target_investments

        inv_return = self.returns.iloc[self.current_index]
        self.previous_wealth = self.wealth
        # w_n = w_n-1 * (1+r)
        self.wealth = np.dot(self.investments, (1 + inv_return))
        #print(self.wealth)

        # todo define new reward function
        reward = self.reward_func(self)

        info = self._get_info()
        state = self._get_state()
        return state, reward, done, info

    def render(self):
        pass

    def reset(self):
        total_index_count = len(self.returns.index)
        last_index = total_index_count-2
        if (self.trade_pecentage >= 1):
            self.start_index = 0
            self.end_index = last_index
        else:
            self.start_index = np.random.randint(low=0, high=last_index*(1-self.trade_pecentage))
            self.end_index = int(self.start_index + total_index_count*self.trade_pecentage)
            self.end_index = min(self.end_index, last_index)
        #print(f'total: {total_index_count}, start index: {self.start_index}, end index: {self.end_index}')
        self.current_index = self.start_index
        self.investments = np.zeros(self.investments_count)
        self.weights = np.zeros(self.investments_count)
        self.previous_wealth=1
        self.wealth = 1
        return self._get_state()

    def _get_state(self):
        state = np.concatenate((self.weights, self.features.iloc[self.current_index]))
        if (state.shape != self.observation_space.shape):
            raise Exception('Shape of state {state.shape} is incorrect should be {self.observation_space.shape}')
        return state

    def _get_info(self):
        start_date = self.returns.index[self.start_index]
        current_date = self.returns.index[self.current_index]
        trade_days = (current_date-start_date).days

        cagr = math.pow(self.wealth, 365/trade_days) - 1
        info = {
            'trade_days': trade_days,
            'wealths': self.wealth,
            'cagr': cagr,
        }
        return info