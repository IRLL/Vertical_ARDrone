# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:19:49 2015

@author: robot3
"""

import numpy as np
from DlogPiDThetaNAC import DlogPiDThetaNAC


def episodicNaturalActorCritic(policy, data, param):

    N = param.param.N
    M = param.param.M
    gamma = param.param.gamma

    # Obtain expected return
    J = 0
    if gamma == 1:
        for Trials in range(np.max(np.shape(data))):
            J = J + np.sum(data[Trials].r) / np.max(np.shape(data[Trials].r))

    # Obtain Gradients
    data_size = np.max(np.shape(data))
    Mat = np.empty(shape=(data_size, (N*M)+1))
    Vec = np.empty(shape=(data_size, 1))
    i = 0
    for Trials in range(data_size):
        x = np.reshape(data[Trials].x[:, 0], (N, 1))
        u = np.reshape(data[Trials].u[:, 0], (M, 1))

        Mat[i, :] = np.append(np.zeros(np.shape(DlogPiDThetaNAC(policy, x, u, param))).reshape(1, N*M), np.array([[1]]), axis=1)
        Vec[i, 0] = 0

        for Steps in range(np.max(np.shape(data[Trials].u))):
            x = np.reshape(data[Trials].x[:, Steps], (N, 1))
            u = np.reshape(data[Trials].u[:, Steps], (M, 1))
            decay_gamma = gamma ** Steps
            Mat[i, :] = Mat[i, :] + decay_gamma * np.append(DlogPiDThetaNAC(policy, x, u, param).reshape(1, N*M), np.array([[0]]), axis=1)
            Vec[i, 0] = Vec[i, 0] + decay_gamma * data[Trials].r[0][Steps] - J
        i = i + 1

    # cond(Mat)
    Nrm = np.diag(np.append(1 / np.std(Mat[:, 0:N*M], ddof=1, axis=0), [1], axis=0))
    mat = Mat
    mat_p = Mat.conj().T
    vec = Vec
    w = np.dot(Nrm, np.dot(np.linalg.inv(np.dot(Nrm, np.dot(mat_p, np.dot(mat, Nrm)))), np.dot(Nrm, np.dot(mat_p, vec))))
    w = w[0:(np.max(np.shape(w))-1)]

    return w.conj().T
