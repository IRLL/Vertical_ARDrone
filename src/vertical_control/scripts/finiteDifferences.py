# -*- coding: utf-8 -*-
"""
@author: david isele
"""

import numpy as np


def finiteDifferences(data, numRollouts):
    dTheta = np.zeros((np.size(data[0].policy.theta), numRollouts-1))
    for h in range((numRollouts-1)/2):
        for p in range( np.size( data[h].policy.theta) ):
            perturb = data[2*h+1].policy.theta[p]-data[0].policy.theta[p]
            #minus = data[2*h+2].policy.theta[p]-data[0].policy.theta[p]
            dTheta[p,h] = perturb
    dTheta = dTheta.T
    #print "dTheta:",np.shape(dTheta), np.shape(np.linalg.pinv(dTheta))

    # change in reward
    dJ = np.zeros((numRollouts-1,1))
    for h in range((numRollouts-1)/2):
        plus = np.mean(data[2*h+1].r)-np.mean(data[0].r)
        minus = np.mean(data[2*h+2].r)-np.mean(data[0].r)
        dJ[h] = plus-minus  # np.sum(data[h+1].r)-np.sum(data[0].r)
    #print "dJ:", np.shape(dJ)

    # Finite Differences
    tmp = np.linalg.pinv(dTheta)
    #print "inv:", np.shape(tmp), "dJ:", np.shape(dJ)
    dJdtheta = tmp.dot(dJ)
    return dJdtheta
