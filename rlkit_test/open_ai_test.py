import gym
from mountain_car_latest import MountainCarEnv
expl_env = gym.make('MountainCarContinuous-v0')

print(dir(MountainCarEnv()))
print(dir(gym.make('MountainCarContinuous-v0')))