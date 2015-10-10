#!/usr/bin/env python

import rospy
from createSys import createSys
from constructPolicies import constructPolicies
from calcThetaStar import calcThetaStar
from initPGELLA import initPGELLA
from learnPGELLA import learnPGELLA
from testPGELLA import testPGELLA
from math import exp

import cPickle as pickle


rospy.init_node('agent_pg_ella', anonymous=False)

nSystems = 2  # Integer number of tasks
learningRate = .01  # Learning rate for stochastic gradient descent


# Parameters for policy
poliType = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                    # 'Gauss' => Gaussian Policy
baseLearner = 'NAC'  # Base Learner
                           # 'REINFORCE' => Episodic REINFORCE
                           # 'NAC' => Episodic Natural Actor Critic
gamma = 0.9  # Discount factor gamma


# Creating Tasks
Tasks = createSys(nSystems, poliType, baseLearner, gamma)


# Constructing policies
Policies = constructPolicies(Tasks)


# Calculating theta
trajLength = 100 # Number of time steps to simulate in the cart-pole system
numRollouts = 50 # Number of trajectories for testing
numIterations = 500 # Number of learning episodes/iterations # 200
Policies = calcThetaStar(Tasks, Policies, learningRate,
                         trajLength, numRollouts, numIterations)



#==========================================================================

# Save PG policies and tasks to a file
pickle.dump(Tasks, open('tasks.p', 'wb'), pickle.HIGHEST_PROTOCOL)
pickle.dump(Policies, open('policies.p', 'wb'), pickle.HIGHEST_PROTOCOL)

Tasks = None
Policies = None

# Load PG policies and tasks to a file
Tasks = pickle.load(open('tasks.p', 'rb'))
Policies = pickle.load(open('policies.p', 'rb'))

#==========================================================================


# Learning the PGELLA

trajLength = 100
numRollouts = 50 # 200
mu1 = exp(-5)  # Sparsity coefficient
mu2 = exp(-5)  # Regularization coefficient
k = 1  # Number of inner layers

modelPGELLA = initPGELLA(Tasks, 1, mu1, mu2, learningRate)

print "Learn PGELLA"
modelPGELLA = learnPGELLA(Tasks, Policies, learningRate,
                          trajLength, numRollouts, modelPGELLA)

# Testing Phase
print "Test Phase"
trajLength = 100
numRollouts = 5 # 100
numIterations = 500 # 200

# Creating new PG policies
PGPol = constructPolicies(Tasks)

# Testing and comparing PG and PG-ELLA
testPGELLA(Tasks, PGPol, learningRate, trajLength,
           numRollouts, numIterations, modelPGELLA)