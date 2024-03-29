# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:20:41 2015

@author: brownyoda
"""

import numpy as np


def DlogPiDThetaNAC(policy, x, u, param):

    # Programmed by Jan Peters
    N = param.param.N
    M = param.param.M

    sigma = np.max(policy.sigma, 0.00001)
    k = policy.theta
    der = np.empty(shape=(0, N))
    for i in range(M):
        der = np.concatenate((der, np.dot((u[i]-np.dot(k[N*(i):N*(i)+N].conj().T, x)), x.conj().T) / (sigma[i]**2)))

    return der