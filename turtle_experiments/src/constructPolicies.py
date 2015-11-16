# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:02:53 2015

@author: brownyoda
"""

import numpy as np
import sys

from structs import PGPolicy, Policy


def constructPolicies(Tasks):
    nSystems = np.shape(Tasks)[0]

    PGPol = [ PGPolicy() for i in range(nSystems) ]

    for i in range(nSystems):
        N = Tasks[i].param.N # Number of states
        M = Tasks[i].param.M # Number of inputs

        poliType = Tasks[i].param.poliType # Structure of policy

        policy = Policy()
        if poliType == 'Gauss':
            #policy.theta = np.random.rand(N * M,1)
            #policy.theta = np.array([[0.5, 0.0, 0.0, .5, 0.0, .5, 0.0, 0.0]]) 
            policy.theta = np.zeros((N * M, 1))
            #policy.theta = np.array([[1,2,3,4,5,6,7,8,9]])
            policy.sigma = np.random.rand(1, M)
            policy.sigma = np.array([[0.001, 0.001, 0.001]])
        else:
            sys.stderr.write("Undefined Policy Type")
            break

        PGPol[i].policy = policy

    return PGPol
