# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:45:38 2015

@author: brownyoda
"""

import numpy as np
import time

from obtainData import obtainData
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic

import matplotlib.pyplot as plt
from matplotlib import cm

def calcThetaStar(Params, Policies, rates, trajlength, rollouts, numIterations):

    plt.ion()

    nSystems = np.shape(Params)[0]
    r = np.empty(shape=(1, trajlength))

    for i in range(nSystems):
        # TODO: Clear screen
        print "@ Task: ", i
        fig = plt.figure(i)
        ax = fig.add_subplot(111)
        ax.grid()

        policy = Policies[i].policy # Resetting policy IMP

        for k in range(numIterations):
            print "@ Iteration: ", k
            print "    Initial Theta: ", policy.theta
            print "    Sigma: ", policy.sigma
            print "    Learning Rate: ", rates
            data = obtainData(policy, trajlength, rollouts, Params[i])
            dJdtheta = None
            if Params[i].param.baseLearner == "REINFORCE":
                dJdtheta = episodicREINFORCE(policy, data, Params[i])
            else:
                dJdtheta = episodicNaturalActorCritic(policy, data, Params[i]) # TODO: won't use but should finish

            policy.theta = policy.theta + rates*dJdtheta.reshape(9, 1)


            for z in range(rollouts):
                r[0, z] = np.sum(data[z].r)

            if np.isnan(r.any()):
                print "System has NAN"
                print "exiting..."
                import sys
                sys.exit(1)

            print "Mean: ", np.mean(r)
            time.sleep(1)
            ax.scatter(k, np.mean(r), marker=u'x', c=np.random.random((2,3)), cmap=cm.jet)
            ax.figure.canvas.draw()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.05)

        Policies[i].policy = policy # Calculating theta
    plt.show(block=True)

    return Policies
