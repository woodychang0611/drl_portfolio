import os
import sys
current_folder = os.path.dirname(__file__)
sys.path.append(os.path.join('current_folder','./..'))

import common 

import numpy as np
import pandas as pd
from itertools import combinations
from pandas import Timestamp
t = Timestamp('2020-01-31')

print (t)
t = common.offset_date(t,2,'weeks')
print(t)