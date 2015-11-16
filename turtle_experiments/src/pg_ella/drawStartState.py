# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 13:02:41 2015

@author: brownyoda
"""

import numpy as np
import sys
def drawStartState(param):
    # Based on Jan Peters' codes

    #N = param.param.N # FIXME: Never used

    mu0 = param.param.mu0
    S0 = param.param.S0
    X0 = np.random.multivariate_normal(mu0.conj().T[0], S0, 1) # FIXME: Not the same as original

    return X0
