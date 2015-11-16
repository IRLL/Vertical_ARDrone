# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:38:10 2015

@author: brownyoda
"""

import numpy as np
import sys

def computeHessian(data, sigma):

    nRollouts = np.shape(data)[0]
    Hes = np.zeros((2, 2))

    for i in range(nRollouts):

        Pos = data[i].x[0, :]
        Reward = data[i].r
        PosSquare = np.sum(Pos ** 2)
        Vel = data[i].x[1, :]
        VelSquare = np.sum(Vel ** 2)
        PosVel = np.sum(Pos * Vel)
        RewardDum = np.sum(Reward)
        Matrix = 1.0 / 1.0 * np.array([[PosSquare, PosVel], [PosVel, VelSquare]]) * RewardDum
        Hes = Hes + Matrix

    Hessian = -Hes * 1.0 / nRollouts

    return Hessian