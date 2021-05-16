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

validate_split_date = Timestamp('2019-03-01')
print(validate_split_date.value)