# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:42:20 2015

@author: robot3
"""

import numpy as np

from DlogPiDThetaREINFORCE import DlogPiDThetaREINFORCE


def episodicREINFORCE(policy, data, param):

    N = param.param.N
    M = param.param.M
    gamma = param.param.gamma

    dJdtheta = np.zeros((np.shape(DlogPiDThetaREINFORCE(policy, np.ones((N,1)), np.ones((M,1)), param))))

    for Trials in range(np.max(np.shape(data))):
        dSumPi = np.zeros((np.shape(DlogPiDThetaREINFORCE(policy, np.ones((N,1)), np.ones((M,1)), param))))
        sumR = 0

        for Steps in range(np.max(np.shape(data[Trials].u))):
            decayGamma = gamma ** Steps
            dSumPi = dSumPi + DlogPiDThetaREINFORCE(policy, np.reshape(data[Trials].x[:,Steps], (N, 1)), np.reshape(data[Trials].u[:,Steps], (M,1)), param)
            sumR = sumR + decayGamma * data[Trials].r[0][Steps]

        dJdtheta = dJdtheta + dSumPi * sumR

    dJdtheta = (1-gamma) * dJdtheta / np.max(np.shape(data))

    return dJdtheta
