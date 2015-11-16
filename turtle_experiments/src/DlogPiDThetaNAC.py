# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:20:41 2015

@author: brownyoda
"""

import numpy as np


def DlogPiDThetaNAC(policy, x, u, param):

    # Programmed by Jan Peters
    N = param.param.N
    M = param.param.M

    sigma = np.max(policy.sigma, 0.00001)
    k = policy.theta.conj().T
    #print ('debug:',k,policy.theta )
    tmp1 = np.shape(k)
    if tmp1[0]==M*N:
        k = k.T

    der = np.empty(shape=(0, N))
    for i in range(M): # for each action
        #der = np.concatenate((der, np.dot((u[i]-np.dot(k[N*(i):N*(i)+N].conj().T, x)), x.conj().T) / (sigma[i]**2)))
        #print ('debug size:',M,N,'=',np.size(k[0, M*(i):M*(i)+N]),k[0, N*(i):N*(i)+N],i )
        u_ = np.dot(k[0, M*(i):M*(i)+N].reshape(1, N), x) ################ M->N
        val = (u[i] - u_) * x / (sigma[i]**2)
        der = np.concatenate((der, val.conj().T))

    
    return der
