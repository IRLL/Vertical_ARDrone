# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:20:16 2015

@author: brownyoda
"""

import numpy as np
from math import sqrt, isinf
import sys


def rewardFnc(x, u):
    print('x:',x)
    rew = -sqrt(np.dot(x.conj().T, x)) - sqrt(np.dot(u.conj().T, u))
    #rew = -sqrt(np.dot(x.conj().T, np.dot(np.eye(np.shape(x)[0]) * 0.00001, x))) \
    #      -sqrt(np.dot(u.conj().T, np.dot(np.eye(np.shape(u)[0]) * 0.00001, u)))
    if isinf(rew):
        print x
        print u
        print "Error: INFINITY"
        sys.exit(1)
    return rew
