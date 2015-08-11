# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:42:20 2015

@author: robot3
"""

import numpy as np

def episodicREINFORCE(policy, data, param):

    N = param.param.N
    M = param.param.M
    gama = param.param.gamma

    #TODO: Check!
    dJdtheta = np.zeros(DlogPiDThetaREINFORCE(policy, np.ones(N,1), np.ones(M,1), param).shape())

    for Trials in range(max(data.shape())):
        dSumPi = np.zeros(DlogPiDThetaREINFORCE(policy, np.ones(N,1), np.ones(M,1), param).shape())
        sumR = 0
        for Steps in range(max((data[Trials].u).shape())):
            dSumPi = dSumPi + DlogPiDThetaREINFORCE(policy, data[Trials].x[:,Steps], data[Trials].u[:,Steps], param)
            sumR = sumR + np.dot(np.linalg.matrix_power(gamma, Steps -1)), data[Trials].r[Steps])
        dJdtheta = dJdtheta + np.dot(dSumPi, sumR)
    dJdTheta = np.dot(1-gamma, dJdtheta) / max(data.shape())

    return dJdTheta
