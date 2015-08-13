# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:20:16 2015

@author: brownyoda
"""

import numpy as np
from math import sqrt, isinf
import sys


def rewardFnc(x, u):
	rew = -sqrt(np.dot(x.conj().T, x)) - sqrt(np.dot(u.conj().T, u))
	if isinf(rew):
		print x
		print u
		print "Error: INFINITY"
		sys.exit(1)
	return rew