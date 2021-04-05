# -*- coding: utf-8 -*-
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


class MarketEnv(gym.Env):
    def __init__(self, investments: DataFrame, features: DataFrame, show_info=False, trade_freq='days',
                 trade_pecentage=0.1):
        self._load_data(investments=investments, features=features, show_info=show_info, trade_freq=trade_freq)
        self._init_action_space()
        self._init_observation_space()
        self.trade_pecentage = trade_pecentage
        self.start_index, self.current_index, self.end_index = 0, 0, 0
        self.seed()
        self.reset()

    def _load_data(self, investments: DataFrame, features: DataFrame, show_info, trade_freq):
        resample_rules = {
            'days': 'D',
            'weeks': 'W',
            'months': 'M'
        }

        if(trade_freq not in resample_rules.keys()):
            raise ValueError(f"trade_freq '{trade_freq}' not supported, must be one of {list(resample_rules.keys())}")

        if investments.isnull().values.any():
            raise ValueError('At least one null value in investments')
        if features.isnull().values.any():
            raise ValueError('At least one null value in investments')
        # make sure dataframes are sorted
        investments = investments.sort_index().sort_index(axis=1)
        features = features.sort_index().sort_index(axis=1)
        # Only keep data as investments
        features = features[(features.index.isin(investments.index))]

        # resample based on trade frequency e.g. weeks or months
        investments = investments.apply(lambda x: 1+x).resample(resample_rules[trade_freq]).prod().apply(lambda x: x-1)
        features = features.apply(lambda x: 1+x).resample(resample_rules[trade_freq]).prod().apply(lambda x: x-1)
        # Scale to -1 and 1
        for col in features.columns:
            features[col] = features[col]/max(abs(features[col].max()), abs(features[col].min()))
        self.features = features
        self.investments = investments

        if show_info:
            sd = investments.index[0]
            ed = investments.index[-1]
            print(f'Trading Frequency: {trade_freq}')
            print(f'{len(self.investments.columns)} investments loaded')
            print(f'{len(self.features.columns)} features loaded')
            print(f'Starts from {sd} to {ed}')

    def _init_action_space(self):
        action_space_size = len(self.investments.columns)
        self.min_action = np.zeros(action_space_size)
        self.max_action = np.ones(action_space_size)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

    def _init_observation_space(self):
        observation_space_size = len(self.investments.columns) + len(self.features.columns)
        self.low_state = np.full(observation_space_size, -1)
        self.high_state = np.ones(observation_space_size)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.current_index > self.end_index:
            raise Exception(f'current_index {current_index} exceed end_index{self.end_index}')
        weights = self._get_weights(action)
        done = True if (self.current_index >= self.end_index) else False

        reward = 1
        state = self._get_state()
        performance = self._get_performance()
        self.current_index += 1

        info = {
            "test": "test",
        }
        return state, reward, done, info

    def render(self):
        pass

    def reset(self):
        total_index_count = len(self.investments.index)
        last_index = total_index_count-1
        if (self.trade_pecentage >= 1):
            self.start_index = 0
            self.end_index = last_index
        else:
            self.start_index = np.random.randint(low=0, high=last_index*(1-self.trade_pecentage))
            self.end_index = int(self.start_index + total_index_count*self.trade_pecentage)
            self.end_index = min(self.end_index, last_index)
        print(f'total: {total_index_count}, start index: {self.start_index}, end index: {self.end_index}')
        self.current_index = self.start_index
        return self._get_state()

    def _get_weights(self, action: np.ndarray):
        if action.sum() == 0:
            action = np.random.rand(*action.shape)
        weight = action / action.sum()
        return weight

    def _get_state(self):
        print(f'Get state for date: {self.investments.index[self.current_index]}')
        return np.zeros(self.observation_space.shape)

    def _get_performance(self):
        start_date = self.investments.index[self.start_index]
        current_date = self.investments.index[self.current_index]
        trade_days = (current_date-start_date).days
        print()
