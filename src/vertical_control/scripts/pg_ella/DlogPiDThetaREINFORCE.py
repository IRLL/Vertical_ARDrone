# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:35:20 2015

@author: robot3
"""

import numpy as np

def DlogPiDThetaREINFORCE(policy, x, u, param):
    N = param.param.N
    M = param.param.M

    sigma = max(policy.sigma, 0.00001)
    k = policy.theta
    xx = x;
    der = []
    for i in range(M):
        #TODO: der = [der;(u(i)-k(1+N*(i-1):N*(i-1)+N)'*xx)*xx/(sigma(i)^2)];

    return der, der2
