# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:12:29 2015

@author: brownyoda
"""

import numpy as np

def drawAction(policy, x, param):

    _theta = policy.theta.reshape(param.param.N, param.param.M)
    _x = x.reshape(3, 1)

    print "Reshape Theta: ", _theta
    print "X: ", _x
    print "Sigma: ", np.diag(policy.sigma[0])
    #print x
    #print policy.sigma
    # Based on Jan Peters' code.
    #u = np.random.multivariate_normal(np.dot(policy.theta.reshape(param.param.N, param.param.M).conj().T, x).conj().T, policy.sigma)

    print "Dot product: ", np.dot(_theta, _x).conj().T[0]
    print
    u = np.random.multivariate_normal(np.dot(_theta, _x).conj().T[0], np.diag(policy.sigma[0]))

    print "ACTION u: ", u
    return u
