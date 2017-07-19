# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 19:24:49 2016

@author: Sergey
"""

import numpy as np
import matplotlib.pyplot as plt
a = np.genfromtxt(r"F:\Program Files\Python27\lossdata.csv",delimiter=",")
a = a.T
plt.plot(a[1], label = "train")
plt.plot(a[2], label = "test")
plt.legend()
plt.ylabel('loss (MSE)')
plt.xlabel('epochs')
plt.show()