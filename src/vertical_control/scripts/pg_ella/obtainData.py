# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:54:40 2015

@author: brownyoda
"""

from structs import Data
from drawStartState import drawStartState
from drawAction import drawAction
from rewardFnc import rewardFnc
from drawNextState import drawNextState

def obtainData(policy, L, H, param):
	N = param.param.N
	M = param.param.M
	data = [Data(N, M, L) for i in range(H)]

	# Based on Jan Peters; codes.
	# Perform H
	for trials in range(H):
		# Save associated policy
		data[trials].policy = policy

		# Draw the first state
		data[trials].x[:,1] = drawStartState(param)

		# Perform a trial of length L
		for steps in range(L):
			# Draw an action from the policy
			data[trials].u[:,steps] = drawAction(policy, data[trials].x[:,steps], param)
			# Obtain the reward from the
			data[trials].r[0][steps] = rewardFnc(data[trials].x[:,steps], data[trials].u[:,steps]) # FIXME: Not similar to original
			# Draw next state from environment
			data[trials].x[:,steps+1] = drawNextState(data[trials].x[:,steps], data[trials].u[:,steps], param, i)

	return data