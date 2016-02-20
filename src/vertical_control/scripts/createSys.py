# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:11:04 2015

@author: brownyoda
"""

import numpy as np

from structs import Param, Tasks


def createSys(nSystems, poliType, baseLearner, gamma):
    tasks = [Tasks() for i in range(nSystems)]

    # Fixed Parameters
    tasks[0].nSystems = nSystems  # Number of tasks

    counter = 1

    for i in range(nSystems):
        param = Param()

        param.N = 3  # Number of states
        param.M = 3  # Number of inputs

        # Calculation of random parameters
        param.disturbance = [0.0] * param.M  # [x, y, z]

        # ensure at least one disturbance is non-zero
        # while param.disturbance[0] == 0.0 and param.disturbance[1] == 0.0:
        # x gets disturbance
        param.disturbance[0] = round(np.random.rand()/2*0.1, 4)
        # y gets disturbance
        param.disturbance[1] = round(np.random.rand()/2*0.1, 4)

        # Parameters for policy
        param.poliType = poliType
        param.baseLearner = baseLearner
        param.gamma = gamma

        # Covariance matrix for Gaussian Policy
        param.S0 = 0.0001 * np.eye(param.N)

        # Assignation of parameters to task i
        param.TaskId = i
        tasks[i].param = param

        counter = counter + 1

    return tasks
