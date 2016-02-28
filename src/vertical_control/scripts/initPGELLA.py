# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 01:39:30 2015

@author: brownyoda
"""

import numpy as np
from structs import Model


def initPGELLA(Tasks, k, mu_one, mu_two, learningRate):

    model = Model()
    model.S = np.zeros((k, np.shape(Tasks)[0]))
    model.mu_one = mu_one
    model.mu_two = mu_two
    model.learningRate = learningRate
    model.k = k

    model.L = np.random.rand(Tasks[0].param.N * Tasks[0].param.M, k)

    return model
