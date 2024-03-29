# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:12:29 2015

@author: brownyoda
"""

import numpy as np

def drawAction(policy, x, param):

    #print policy.theta
    #print x
    #print policy.sigma
    # Based on Jan Peters' code.
    u = np.random.multivariate_normal(np.dot(policy.theta.reshape(param.param.N, param.param.M).conj().T, x).conj().T, policy.sigma)

    return u
