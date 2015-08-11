# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:32:05 2015

@desc: Creating Systems and calculating learning rates
"""

from createSys import createSys
from constructPolicies import constructPolicies

nSystems = 10 # Integer number of tasks
learningRate = .3 # Learning rate for stochastic gradient descent


# Parameters for policy
poliType = 'Gauss' # Policy Type (Only supports Gaussian Policies)
				# 'Gauss' => Gaussian Policy
baseLearner = 'NAC' # Base Learner
				# 'REINFORCE' => Episodic REINFORCE
				# 'NAC' => Episodic Natural Actor Critic
gamma = 0.9 # Discount factor gamma


# Creating Tasks
Tasks = createSys(nSystems, poliType, baseLearner, gamma)


# Constructing policies
[Policies] = constructPolicies(Tasks)

'''
# Calculating theta
trajLength = 10
numRollouts = 100
numIterations = 200
[Policies] = calcThetaStar(Tasks, Policies, learningRate, trajLength, numRollouts, numIterations)


# Learning the PGELLA

trajLength = 10
numRollouts = 100
mu1 = exp(-5) # Sparsity coefficient
mu2 = exp(-5) # Regularization coefficient
k = 1 # Number of inner layers

[modelPGELLA] = initPGELLA(Tasks, 1, mu1, mu2, learningRate)

[modelPGELLA] = learnPGELLA(Tasks, Policies, learningRate, trajLength, numRollouts, modelPGELLA)

# Testing Phase

trajLength = 10
numRollouts = 100
numIterations = 200

# Creating new PG policies
[PGPol] = constructPolicies(Tasks)

# Testing and comparing PG and PG-ELLA
testPGELLA(Tasks, PGPol, learningRate, trajLength, numRollouts, numIterations, modelPGELLA)
'''