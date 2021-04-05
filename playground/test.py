import os
import sys
current_folder = os.path.dirname(__file__)
sys.path.append(os.path.join('current_folder','./..'))

import common 

import numpy as np
import pandas as pd
from itertools import combinations
from pandas import Timestamp



action = np.concatenate( (np.zeros(3),np.ones(5)) )
print(action)