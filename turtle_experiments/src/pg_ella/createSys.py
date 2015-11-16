# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:11:04 2015

@author: brownyoda
"""

import numpy as np

from structs import Param, Tasks


def createSys(nSystems, poliType, baseLearner, gamma):
    tasks = [ Tasks() for i in range(nSystems) ]

    # Fixed Parameters
    tasks[0].nSystems = nSystems # Number of tasks

    # Range of parameters for dynamical systems
    tasks[0].MassMin = 3
    tasks[0].MassMax = 5
    tasks[0].Mink = 1
    tasks[0].Maxk = 7
    tasks[0].Maxd = 0.1
    tasks[0].Mind = 0.01

    counter = 1

    for i in range(nSystems):
        param = Param()

        param.N = 2 # Number of states
        param.M = 1 # Number of inputs

        # Differential for integration of dynamical system
        param.dt = .01

        # Calculation of random parameters
        param.Mass = (tasks[0].MassMax - tasks[0].MassMin) * np.random.rand() + tasks[0].MassMin
        param.k = (tasks[0].Maxk - tasks[0].Mink) * np.random.rand() + tasks[0].Mink
        param.d = (tasks[0].Maxd - tasks[0].Mind) * np.random.rand() + tasks[0].Mind

        # Initial and reference (final) states
        param.mu0 = 2 * np.random.rand(2, 1)
        param.Xref = np.zeros((2, 1))

        # Parameters for policy
        param.poliType = poliType
        param.baseLearner = baseLearner
        param.gamma = gamma

        # Covariance matrix for Gaussian Policy
        param.S0 = 0.0001 * np.eye(param.N) # Covariance matrix for Gaussian Policy # FIXME

        # Assignation of parameters to task i
        param.TaskId = i
        tasks[i].param = param

        counter = counter + 1

    return tasks
