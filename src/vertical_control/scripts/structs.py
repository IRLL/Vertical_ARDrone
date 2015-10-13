# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:46:46 2015

@author: robot3
"""

import numpy as np


class Data():
    def __init__(self, n, m, traj_length):
        self.policy = None
        self.x = np.zeros(shape=(n, traj_length + 1)) * np.nan
        self.u = np.zeros(shape=(m, traj_length)) * np.nan
        '''self.u = []
        for i in range(m):
            self.u.append(np.empty(shape=(1,traj_length)))'''
        self.r = np.zeros(shape=(1, traj_length)) * np.nan


class Param():

    def __init__(self):
        self.N = 0
        self.M = 0
        self.dt = 0
        self.Mass = 0
        self.k = 0
        self.d = 0
        self.mu0 = None
        self.Xref = None
        self.poliType = None
        self.baseLearner = None
        self.gamma = None
        self.S0 = None
        self.TaskId = 0


class Tasks():

    nSystems = 0
    MassMin = 0
    MassMax = 0
    Mink = 0
    Maxk = 0
    Maxd = 0
    Mind = 0

    def __init__(self):
        self.param = None


class PGPolicy:

    def __init__(self):
        self.policy = None


class Policy:

    def __init__(self):
        self.theta = None
        self.sigma = None


class Model:

    def __init__(self):
        self.S = None
        self.mu_one = None
        self.mu_two = None
        self.learningRate = 0.0
        self.L = None


class Hessianarray:

    def __init__(self):
        self.D = None


class Parameterarray:

    def __init__(self):
        self.alpha = None