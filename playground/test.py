import os
import sys
import math
from pandas.io.pytables import IndexCol
current_folder = os.path.dirname(__file__)
sys.path.append(os.path.join('current_folder','./..'))

import common 

import numpy as np
import pandas as pd
from itertools import combinations
from pandas import Timestamp
from common.finance_utility import finance_utility
import matplotlib.pyplot as plt

def get_func(a):
    def F(s):
        print(f'{a}:{s}')
    return F


f = get_func('test')
kwargs = dict(s='1234')
f(**kwargs)