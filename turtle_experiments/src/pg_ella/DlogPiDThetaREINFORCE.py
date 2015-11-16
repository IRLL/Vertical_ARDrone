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
    k = policy.theta

    der = np.empty(shape=(0, N))
    for i in range(M):
        der = np.concatenate((der, np.dot((u[i]-np.dot(k[N*(i):N*(i)+N].conj().T, x)), x.conj().T) / (sigma[i]**2)))

    return der