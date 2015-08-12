# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:26:52 2015

@author: brownyoda
"""

import numpy as np


def drawNextState(x, u, param, i):
	# Based on Jan Peters' code.

	A = np.array([ [0                              , 1],
                    [-param.param.k/param.param.Mass, - param.param.d/param.param.Mass] ])
	b = np.array([ [0], [1.0/param.param.Mass] ])
	xd = np.dot(A, x) + np.dot(b, u)
	xn = x + np.dot(param.param.dt, xd)

	return xn
