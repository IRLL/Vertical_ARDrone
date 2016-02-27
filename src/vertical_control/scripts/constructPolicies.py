# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:02:53 2015

@author: brownyoda
"""

import numpy as np
import sys
import types

from structs import PGPolicy, Policy


def constructPolicies(Tasks, init_policy=None):
    nSystems = np.shape(Tasks)[0]

    PGPol = [PGPolicy() for i in range(nSystems)]

    for i in range(nSystems):
        N = Tasks[i].param.N  # Number of states
        M = Tasks[i].param.M  # Number of inputs

        poliType = Tasks[i].param.poliType  # Structure of policy

        policy = Policy()
        if poliType == 'Gauss':
            # policy.theta = np.random.rand(N * M,1) * 0.1
            policy.theta = np.zeros((N * M, 1))
            # for j in range(N*M):
            #    val = policy.theta[j]
            #    policy.theta[j] = val if np.random.randint(2) == 1 else -val
            if not isinstance(init_policy, types.NoneType):
                policy.theta = init_policy
            # policy.sigma = np.random.rand(1, M)
            policy.sigma = np.array([[0, 0, 0]])
        else:
            sys.stderr.write("Undefined Policy Type")
            break

        PGPol[i].policy = policy

    return PGPol
