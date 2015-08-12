# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:45:38 2015

@author: brownyoda
"""

import numpy as np

from obtainData import obtainData
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
import sys

def calcThetaStar(Params, Policies, rates, trajlength, rollouts, numIterations):

	nSystems = np.shape(Params)[0]

	for i in range(nSystems):
		# TODO: Clear screen

		policy = Policies[i].policy # Resetting policy IMP
		for k in range(numIterations):
			print "@ Iteration: ", k
			data = obtainData(policy, trajlength, rollouts, Params[i])
			dJdtheta = None
			if Params[i].param.baseLearner == "REINFORCE":
				dJdtheta = episodicREINFORCE(policy, data, Params[i])
			else:
				dJdtheta = episodicNaturalActorCritic(policy, data, Params[i]) # TODO: won't use but should finish

			policy.theta = policy.theta + rates*dJdtheta.conj().T

		Policies[i].policy = policy # Calculating theta
		print "Policy ", i, ": ", policy.theta
	return Policies