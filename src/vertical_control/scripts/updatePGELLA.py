# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:50:37 2015

@author: brownyoda
"""
from __future__ import print_function
import numpy as np
import spams
from numpy.lib.scimath import sqrt as csqrt


def updatePGELLA(ELLAmodel, taskId, ObservedTasks,
                 HessianArray, ParameterArray):

    # ------------------------------------------------------------------------
    # Update L -- Tasks[taskId].param.Group --> Know which Group ..
    # ------------------------------------------------------------------------
    summ = np.zeros(np.shape(ELLAmodel.L))
    allowedTask = np.nonzero(ObservedTasks == 1)[0]
    Tg = np.sum(ObservedTasks)
    k = ELLAmodel.k
    for i in range(int(Tg)):
        # print(np.shape(ELLAmodel.S[:,allowedTask[i]]))
        S = ELLAmodel.S[:, allowedTask[i]].reshape(np.size(ELLAmodel.S[:, allowedTask[i]]), 1)
        S_p = S.conj().T
        # print "shape 1: ", np.shape((2 * HessianArray[allowedTask[i]].D))
        # print "shape 2: ", np.shape(np.dot(ParameterArray[allowedTask[i]].alpha, S_p))

        # print "2nd term:", np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, S_p)).reshape(np.shape(ELLAmodel.L))
        tmp2 = np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, S_p)).reshape(np.shape(ELLAmodel.L))

        # print "3rd term:", np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T)))
        tmp3 = np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T)))

        # print "4th term:", 2 * ELLAmodel.mu_two * ELLAmodel.L
        tmp4 = 2 * ELLAmodel.mu_two * ELLAmodel.L
        summ = summ - tmp2 + tmp3 + tmp4
        # summ = summ - np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, ELLAmodel.S[:,allowedTask[i]].conj().T)).reshape(np.shape(ELLAmodel.L)) \
        #            + np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T))) \
        #            + 2 * ELLAmodel.mu_two * ELLAmodel.L

    ELLAmodel.L = (ELLAmodel.L - ELLAmodel.learningRate * 1.0 / Tg * summ)
    # ------------------------------------------------------------------------
    # Update s_{taskId} using LASSO
    # ------------------------------------------------------------------------
    # Determine which group taskId belongs to
    Dsqrt = HessianArray[taskId].D ** .5
    # Dsqrt = Dsqrt.real
    target = np.dot(Dsqrt, ParameterArray[taskId].alpha)
    dictTransformed = np.dot(Dsqrt, ELLAmodel.L)

    '''
    lasso(X,D= None,Q = None,q = None,return_reg_path = False,
          L= -1,lambda1=None,lambda2= 0.,
          mode= spams_wrap.PENALTY,pos= False,ols= False,numThreads= -1,
          max_length_path= -1,verbose=False,cholesky= False):
    '''
    s = spams.lasso(np.asfortranarray(target, dtype=np.float64),
                    D=np.asfortranarray(dictTransformed, dtype=np.float64),
                    Q=None, q=None, return_reg_path=False, L=-1,
                    lambda1=ELLAmodel.mu_one / 2.0,
                    lambda2=0., verbose=False, mode=2)

    # print(ELLAmodel.S[:, taskId])
    # print(np.asarray(s.todense()).T[0])
    ELLAmodel.S[:, taskId] = np.asarray(s.todense()).T[0]

    print("ELLA Model S: ", ELLAmodel.S)
    print("Theta (L*S): ", np.dot(ELLAmodel.L, ELLAmodel.S[:, taskId].reshape(k, 1)).T)

    return ELLAmodel
# def updatePGELLA(ELLAmodel, taskId, ObservedTasks, HessianArray, ParameterArray):
#
#     #------------------------------------------------------------------------
#     # Update L -- Tasks[taskId].param.Group --> Know which Group ..
#     #------------------------------------------------------------------------
#     summ = np.zeros(np.shape(ELLAmodel.L))
#     allowedTask = np.nonzero(ObservedTasks == 1)[0]
#     Tg = np.sum(ObservedTasks)
#
#     k = ELLAmodel.k
#     for i in range(int(Tg)):
#         S = ELLAmodel.S[:,allowedTask[i]].reshape(k,1)
#         S_p = S.conj().T
#         D = HessianArray[allowedTask[i]].D
#         #print "Task ", i, " D: ", D
#
#         summ = summ - np.dot(np.dot((2 * D), ParameterArray[allowedTask[i]].alpha), S_p).reshape(np.shape(ELLAmodel.L)) \
#                     + np.dot(np.dot(np.dot((2 * D), ELLAmodel.L), S), S_p) \
#                     + 2 * ELLAmodel.mu_two * ELLAmodel.L
#
#         #summ = summ - np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, S_p)).reshape(np.shape(ELLAmodel.L)) \
#         #            + np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(S, S_p))) \
#         #            + 2 * ELLAmodel.mu_two * ELLAmodel.L
#         #summ = summ - np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ParameterArray[allowedTask[i]].alpha, ELLAmodel.S[:,allowedTask[i]].conj().T)).reshape(np.shape(ELLAmodel.L)) \
#         #            + np.dot((2 * HessianArray[allowedTask[i]].D), np.dot(ELLAmodel.L, np.dot(ELLAmodel.S[:,allowedTask[i]], ELLAmodel.S[:,allowedTask[i]].conj().T))) \
#         #            + 2 * ELLAmodel.mu_two * ELLAmodel.L
#
#     ELLAmodel.L = (ELLAmodel.L - ELLAmodel.learningRate * ((1 / Tg) * summ))
#
#     print("ELLA Model L: ", ELLAmodel.L)
#
#     #------------------------------------------------------------------------
#     # Update s_{taskId} using LASSO
#     #------------------------------------------------------------------------
#     # Determine which group taskId belongs to
#     Dsqrt = csqrt(HessianArray[taskId].D)
#     Dsqrt = Dsqrt.real # get the real parts of all elements in the array
#     target = np.dot(Dsqrt, ParameterArray[taskId].alpha)
#     dictTransformed = np.dot(Dsqrt, ELLAmodel.L)
#
#     '''
#     lasso(X,D= None,Q = None,q = None,return_reg_path = False,L= -1,lambda1= None,lambda2= 0.,
#                  mode= spams_wrap.PENALTY,pos= False,ols= False,numThreads= -1,
#                  max_length_path= -1,verbose=False,cholesky= False):
#     '''
#     s = spams.lasso(np.asfortranarray(target, dtype=np.float64), D = np.asfortranarray(dictTransformed, dtype=np.float64),
#                             Q = None, q = None, return_reg_path = False, L = -1, lambda1 = ELLAmodel.mu_one / 2.0, lambda2 = 0., verbose = False, mode = 2)
#
#     ELLAmodel.S[:, taskId] = np.asarray(s.todense())
#
#     print("ELLA Model S: ", ELLAmodel.S)
#     print("Theta (L * S): ", np.dot(ELLAmodel.L, ELLAmodel.S[:, taskId].reshape(k, 1)))
#
#     return ELLAmodel
