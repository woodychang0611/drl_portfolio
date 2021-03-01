import numpy as np
from math import dist
from my_gym_env import MyGymEnv
from random import uniform
print(uniform(-10,10))

env = MyGymEnv()
env.render()
for i in range(0,10):
    action = np.array([1,1])
    s,r,d,_ = env.step(action)
    print(d)
    env.render()



