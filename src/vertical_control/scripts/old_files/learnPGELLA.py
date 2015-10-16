# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:45:30 2015

@author: brownyoda
"""

import numpy as np

from obtainData import obtainData
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
from computeHessian import computeHessian
from updatePGELLA import updatePGELLA
from structs import Hessianarray, Parameterarray


def learnPGELLA(Tasks, Policies, learningRate,
                trajLength, numRollouts, modelPGELLA):

    counter = 1
    tasks_size = np.shape(Tasks)[0]
    ObservedTasks = np.zeros((tasks_size, 1))
    limitOne = 0
    limitTwo = tasks_size


    HessianArray = [Hessianarray() for i in range(tasks_size)]
    ParameterArray = [Parameterarray() for i in range(tasks_size)]

    while not np.all(ObservedTasks):  # Repeat until all tasks are observed

        taskId = np.random.randint(limitOne, limitTwo, 1)  # Pick a random task
        print "Task ID: ", taskId

        if ObservedTasks[taskId] == 0:  # Entry is set to 1 when corresponding task is observed
            ObservedTasks[taskId] = 1
    #----------------------------------- ------------------------------------
    # Policy Gradientts on taskId
    #------------------------------------------------------------------------
        data = obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

        if Tasks[taskId].param.baseLearner == 'REINFORCE':
            dJdTheta = episodicREINFORCE(Policies[taskId].policy, data, Tasks[taskId])
        else:
            dJdTheta = episodicNaturalActorCritic(Policies[taskId].policy, data, Tasks[taskId])

        # Updating theta*
        Policies[taskId].policy.theta = Policies[taskId].policy.theta + learningRate * dJdTheta.reshape(9, 1)

    #------------------------------------------------------------------------

        # Computing Hessian
        data = obtainData(Policies[taskId].policy, trajLength, numRollouts, Tasks[taskId])

        D = computeHessian(data, Policies[taskId].policy.sigma)
        HessianArray[taskId].D =  D

        ParameterArray[taskId].alpha = Policies[taskId].policy.theta

    #------------------------------------------------------------------------
        # Perform Updating L and S
        modelPGELLA = updatePGELLA(modelPGELLA, taskId, ObservedTasks, HessianArray, ParameterArray)  # Perform PGELLA for that Group
    #------------------------------------------------------------------------

        print 'Iterating @: ', counter
        counter = counter + 1

    print 'All Tasks observed @: ', counter-1

    return modelPGELLA
