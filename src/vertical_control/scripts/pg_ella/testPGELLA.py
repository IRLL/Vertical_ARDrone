# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:09:15 2015

@author: brownyoda
"""

import numpy as np
from structs import PGPolicy, Policy
from obtainData import obtainData
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic

import matplotlib.pyplot as plt
from matplotlib import cm


def testPGELLA(Tasks, PGPol, learningRate, trajLength,
               numRollouts, numIterations, modelPGELLA):

    plt.ion()

    tasks_size = np.shape(Tasks)[0]

    PolicyPGELLAGroup = [PGPolicy() for i in range(tasks_size)]

    for i in range(tasks_size):  # loop over all tasks
        theta_PG_ELLA = modelPGELLA.L * modelPGELLA.S[:,i] # TODO: double check
        policyPGELLA = Policy()
        policyPGELLA.theta = theta_PG_ELLA
        policyPGELLA.sigma = PGPol[i].policy.sigma
        PolicyPGELLAGroup[i].policy = policyPGELLA

    Avg_rPGELLA = np.zeros((numIterations, tasks_size))
    Avg_rPG = np.zeros((numIterations, tasks_size))
    for k in range(tasks_size): # Test over all tasks
        print "@ Task: ", k
        fig = plt.figure(k)
        ax = fig.add_subplot(111)
        ax.grid()
        for m in range(numIterations): # Loop over Iterations
            print "    @ Iteration: ", m
            # PG
            data = obtainData(PGPol[k].policy, trajLength, numRollouts, Tasks[k])

            if Tasks[k].param.baseLearner == 'REINFORCE':
                dJdTheta = episodicREINFORCE(PGPol[k].policy, data, Tasks[k])
            else:
                dJdTheta = episodicNaturalActorCritic(PGPol[k].policy, data, Tasks[k])

            # Update Policy Parameters
            PGPol[k].policy.theta = PGPol[k].policy.theta + learningRate * dJdTheta.conj().T
            print "PG policy: ", PGPol[k].policy.theta

            # PG-ELLA
            dataPGELLA = obtainData(PolicyPGELLAGroup[k].policy, trajLength, numRollouts, Tasks[k])

            if Tasks[k].param.baseLearner == 'REINFORCE':
                dJdThetaPGELLA = episodicREINFORCE(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])
            else:
                dJdThetaPGELLA = episodicNaturalActorCritic(PolicyPGELLAGroup[k].policy, dataPGELLA, Tasks[k])

            # Update Policy Parameters
            PolicyPGELLAGroup[k].policy.theta = PolicyPGELLAGroup[k].policy.theta + learningRate * dJdThetaPGELLA.conj().T
            print "PGELLA policy: ", PolicyPGELLAGroup[k].policy.theta

            # Computing Average in one System per iteration
            data_size = np.shape(data)[0]
            Sum_rPG = np.zeros((data_size, 1))
            for z in range(data_size):
                Sum_rPG[z,:] = np.sum(data[z].r)

            dataPG_size = np.shape(dataPGELLA)[0]
            Sum_rPGELLA = np.zeros((dataPG_size, 1))
            for z in range(dataPG_size):
                Sum_rPGELLA[z,:] = np.sum(dataPGELLA[z].r)

            Avg_rPGELLA[m, k] = np.mean(Sum_rPGELLA)
            Avg_rPG[m, k] = np.mean(Sum_rPG)

            # TODO: Plot graph
            ax.scatter(m, Avg_rPGELLA[m, k], marker=u'*', color='r', cmap=cm.jet)
            ax.scatter(m, Avg_rPG[m, k], marker=u'*', color='b', cmap=cm.jet)
            ax.figure.canvas.draw()
            #fig.canvas.blit(ax.bbox)
            #fig.canvas.draw()
            fig.canvas.draw()
            fig.canvas.flush_events()
    plt.show(block=True)