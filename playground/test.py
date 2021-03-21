import numpy as np
import pandas as pd

s = pd.Series()
print(type(s))

def a(self):
    print('a')

pd.Series.a = a
s.a()