# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:35:20 2015

@author: robot3
"""

import numpy as np


def DlogPiDThetaREINFORCE(policy, x, u, param):
    N = param.param.N
    M = param.param.M

    sigma = np.max(policy.sigma, 0.00001)
    k = policy.theta.conj().T

    der = np.empty(shape=(0, N))
    for i in range(M):
        u_ = np.dot(k[0, N*(i):N*(i)+N].reshape(1, M), x)
        val = (u[i] - u_) * x / (sigma[i]**2)
        der = np.concatenate((der, val.conj().T))
    return der
