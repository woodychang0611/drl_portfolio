# -*- coding: utf-8 -*-
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


class MarketEnv(gym.Env):

    def load_data(self, investments: DataFrame, features: DataFrame, show_info, trade_freq):
        sample_rules = {
            'days': 'D',
                    'weeks': 'W',
                    'months': 'M'
        }

        if(trade_freq not in sample_rules.keys()):
            raise ValueError(f"trade_freq '{trade_freq}' not supported, must be one of {list(sample_rules.keys())}")

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
        investments = investments.apply(lambda x: 1+x).resample(sample_rules[trade_freq]).prod().apply(lambda x: x-1)
        features = features.apply(lambda x: 1+x).resample(sample_rules[trade_freq]).prod().apply(lambda x: x-1)
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

    def init_action_space(self):
        action_space_size = len(self.investments.columns)
        self.min_action = np.zeros(action_space_size)
        self.max_action = np.ones(action_space_size)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

    def init_observation_space(self):
        observation_space_size = len(self.investments.columns) + len(self.features.columns)
        self.low_state = np.full(observation_space_size, -1)
        self.high_state = np.ones(observation_space_size)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    def __init__(self, investments: DataFrame, features: DataFrame, show_info=False, trade_freq='days',
                 fix_start_time=False, min_trade_pecentage=0.1):
        self.load_data(investments=investments, features=features, show_info=show_info, trade_freq=trade_freq)
        self.init_action_space()
        self.init_observation_space()
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        state, reward, done = 1, 1, 1, 1
        return state, reward, done, {}

    def render(self):
        pass

    def reset(self):
        state = 1
        return state
