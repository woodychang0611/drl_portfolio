# -*- coding: utf-8 -*-
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


class MarketEnv(gym.Env):
    def __init__(self, investments: DataFrame, features: DataFrame, show_info=False, trade_freq='days'):
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
            print(f'{len(investments.columns)} investments loaded')
            print(f'{len(features.columns)} features loaded')
            print(f'Starts from {sd} to {ed}')

        self.max_step_distance = 1
        self.goal_distance = 0.1
        self.total_distance = 10
        self.min_action = np.array([-self.max_step_distance, -self.max_step_distance])
        self.max_action = np.array([self.max_step_distance, self.max_step_distance])
        self.low_state = np.array([-self.total_distance, -self.total_distance])
        self.high_state = np.array([self.total_distance, self.total_distance])
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        reward = action
        old_dist = np.linalg.norm(self.state)
        self.state += action
        dist = np.linalg.norm(self.state)
        if (np.greater(abs(self.state), self.total_distance).any()):
            done = True
            reward = -100
        elif (dist <= self.goal_distance):
            done = True
            reward = 100
        else:
            done = False
            reward = old_dist-dist-0.2
        return self.state, reward, done, {}

    def render(self):
        print(f'state is {self.state}')

    def reset(self):
        self.state = np.array([uniform(-self.total_distance, self.total_distance),
                               uniform(-self.total_distance, self.total_distance)])
        return np.array(self.state)
