# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:12:29 2015

@author: brownyoda
"""

import numpy as np

def drawAction(policy, x, param):

    #print('debug:', np.shape(policy.theta),policy.theta )

    _theta = policy.theta.reshape(param.param.N, param.param.M)

    _x = x.reshape(param.param.N, 1) #################### this was wrong

    _u = np.zeros((1, param.param.M))
    for i in range(param.param.M):
        mean = np.dot(_theta[:,i], _x)
        cov = np.array([[policy.sigma[0, i]]])
        _u[0, i] = np.random.multivariate_normal(mean, cov)

    return _u[0]
