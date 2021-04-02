import numpy as np
from math import dist
from my_gym_env import MyGymEnv
from random import uniform
import pandas as pd

def test():
    n1 = 1
    def test2():
        nonlocal  n1
        print(n1)
        n1 = 123
    test2()
    print(n1)

test()
