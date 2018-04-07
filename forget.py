# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 19:17:10 2018

@author: user
"""

import numpy as np

np.random.seed(42)
out = []
for i in range(20):
    np.random.seed(42)
    randomize = np.arange(100)
    np.random.shuffle(randomize)
    out.append(randomize)
