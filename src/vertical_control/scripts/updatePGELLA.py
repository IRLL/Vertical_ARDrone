# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:50:37 2015

@author: brownyoda
"""

import numpy as np
#from sklearn.linear_model import LassoLars
import scipy
import spams
import sys

def updatePGELLA(ELLAmodel, taskId, ObservedTasks, HessianArray, ParameterArray):

    #------------------------------------------------------------------------
    # Update L -- Tasks[taskId].param.Group --> Know which Group ..
    #------------------------------------------------------------------------
    summ = np.zeros(np.shape(ELLAmodel.L))
    #allowedTask = np.where(ObservedTasks == 1)[0]
    allowedTask = np.nonzero(ObservedTasks == 1)[0]
    Tg = np.sum(ObservedTasks)

    print "MU2:",ELLAmodel.mu_two
    print "ELLA L:",ELLAmodel.L
    for i in range(int(Tg)):
        print "summ:", summ
        print "Hessian D:", HessianArray[allowedTask[i]].D
        print "alpha:", ParameterArray[allowedTask[i]].alpha
        print "L:", ELLAmodel.L
        print "allowedTask:", allowedTask[i]
        print "S:", ELLAmodel.S
        print "S:", ELLAmodel.S[:,allowedTask[i]].conj().T
        S = ELLAmodel.S[:,allowedTask[i]].reshape(1,1)
        S_p = S.conj().T
        print "shape 1: ", np.shape((2 * HessianArray[allowedTask[i]].D))
        print "shape 2: ", np.shape(np.dot(ParameterArray[allowedTask[i]].alpha, S_p))
        print "2nd term:", np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, S_p)).reshape(np.shape(ELLAmodel.L))
        print "3rd term:", np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T)))
        print "4th term:", 2 * ELLAmodel.mu_two * ELLAmodel.L
        summ = summ - np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, ELLAmodel.S[:,allowedTask[i]].conj().T)).reshape(np.shape(ELLAmodel.L)) \
                    + np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T))) \
                    + 2 * ELLAmodel.mu_two * ELLAmodel.L

    ELLAmodel.L = (ELLAmodel.L - ELLAmodel.learningRate * 1.0 / Tg * summ)
    #------------------------------------------------------------------------
    # Update s_{taskId} using LASSO
    #------------------------------------------------------------------------
    # Determine which group taskId belongs to
    Dsqrt = HessianArray[taskId].D ** .5
    # Dsqrt = sqrtm(HessianArray[idxTask].D)
    target = np.dot(Dsqrt, ParameterArray[taskId].alpha)
    dictTransformed = np.dot(Dsqrt, ELLAmodel.L)

    '''
    lasso(X,D= None,Q = None,q = None,return_reg_path = False,L= -1,lambda1= None,lambda2= 0.,
                 mode= spams_wrap.PENALTY,pos= False,ols= False,numThreads= -1,
                 max_length_path= -1,verbose=False,cholesky= False):
    '''
    s = spams.lasso(np.asfortranarray(target, dtype=np.float64), D = np.asfortranarray(dictTransformed, dtype=np.float64),
                            Q = None, q = None, return_reg_path = False, L = -1, lambda1 = ELLAmodel.mu_one / 2.0, lambda2 = 0., verbose = False, mode = 2)
    #print "s array: ", s.toarray() #, "path: ", path
    print "s dense: ", s.todense()
    #print np.shape(ELLAmodel.S[:, taskId])
    #print np.shape(np.asarray(s.todense()))
    ELLAmodel.S[:, taskId] = np.asarray(s.todense())

    print "ELLA Model: ", ELLAmodel.S

    return ELLAmodel