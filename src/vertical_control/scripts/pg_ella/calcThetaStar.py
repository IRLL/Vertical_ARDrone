# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:15:20 2015

@author: robot3
"""

import numpy as np

def calcThetaStar(Params, Policies, rates, trajlength, rollouts, numIterations):
    for i in Params:
        policy = Policies[i].policy
        for k in range(numIterations):
            print "@ Iteration ", k
            data = obtainData(policy, trajlength,rollouts,Params(i));
            dJdtheta = 0 #Keep the variable in scope for later use
            if Params[i].param.baseLearner == "REINFORCE":
                dJdtheta = episodicREINFORCE(policy, data, Params[i])
            else:
                dJdtheta = episodicNaturalActorCritic(policy, data, Params[i])
            policy.theta = policy.theta + np.dot(rates, dJdtheta)
        Policies[i].policy = policy
    return Policies
