# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:38:10 2015

@author: brownyoda
"""

import numpy as np


def computeHessian(data, sigma):

    nRollouts = np.shape(data)[0]
    Hes = np.zeros((9, 9))

    for i in range(nRollouts):
        Reward = data[i].r
        XPos = data[i].x[0, :]
        XPosSquare = np.sum(XPos ** 2)
        YPos = data[i].x[1, :]
        YPosSquare = np.sum(YPos ** 2)
        ZPos = data[i].x[2, :]
        ZPosSquare = np.sum(ZPos ** 2)

        XPosYPos = np.sum(XPos * YPos)
        XPosZPos = np.sum(XPos * ZPos)
        YPosZPos = np.sum(YPos * ZPos)

        RewardDum = np.sum(Reward)
        Array = np.array([[XPosSquare, XPosSquare, XPosSquare,   XPosYPos,   XPosYPos,   XPosYPos,   XPosZPos,   XPosZPos,   XPosZPos],
                          [XPosSquare, XPosSquare, XPosSquare,   XPosYPos,   XPosYPos,   XPosYPos,   XPosZPos,   XPosZPos,   XPosZPos],
                          [XPosSquare, XPosSquare, XPosSquare,   XPosYPos,   XPosYPos,   XPosYPos,   XPosZPos,   XPosZPos,   XPosZPos],
                          [  XPosYPos,   XPosYPos,   XPosYPos, YPosSquare, YPosSquare, YPosSquare,   YPosZPos,   YPosZPos,   YPosZPos],
                          [  XPosYPos,   XPosYPos,   XPosYPos, YPosSquare, YPosSquare, YPosSquare,   YPosZPos,   YPosZPos,   YPosZPos],
                          [  XPosYPos,   XPosYPos,   XPosYPos, YPosSquare, YPosSquare, YPosSquare,   YPosZPos,   YPosZPos,   YPosZPos],
                          [  XPosZPos,   XPosZPos,   XPosZPos,   YPosZPos,   YPosZPos,   YPosZPos, ZPosSquare, ZPosSquare, ZPosSquare],
                          [  XPosZPos,   XPosZPos,   XPosZPos,   YPosZPos,   YPosZPos,   YPosZPos, ZPosSquare, ZPosSquare, ZPosSquare],
                          [  XPosZPos,   XPosZPos,   XPosZPos,   YPosZPos,   YPosZPos,   YPosZPos, ZPosSquare, ZPosSquare, ZPosSquare]])
        Matrix = 1. / 1 *(Array * RewardDum)
        #Matrix = 1.0 / 1.0 * np.array([[PosSquare, PosVel], [PosVel, VelSquare]]) * RewardDum
        Hes = Hes + Matrix

    Hessian = -Hes * (1. / nRollouts)

    return Hessian
