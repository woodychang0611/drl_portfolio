# -*- coding: utf-8 -*-
import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
from random import uniform

class MyGymEnv(gym.Env):
    def __init__(self):
        print("MyGymEnv")
        self.max_step_distance = 1
        self.goal_distance=0.1
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
        if (np.greater(abs(self.state),self.total_distance).any()):
            done = True
            reward = -100
        elif (dist <= self.goal_distance):
            done=True
            reward=100
        else:
            done =False
            reward=old_dist-dist-0.2
        return self.state, reward, done, {}

    def render(self):
        print(f'state is {self.state}')
    def reset(self):
        self.state = np.array([uniform(-self.total_distance,self.total_distance),
            uniform(-self.total_distance,self.total_distance)])
        return np.array(self.state)

