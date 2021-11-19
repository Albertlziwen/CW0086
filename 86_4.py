# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-

"""
    File name: ssm_kalman.py
    Description: a re-implementation of the Kalman filter for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Author: Roman Pogodin / Maneesh Sahani (matlab version)
    Date created: October 2018
    Python version: 3.6
"""


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth'):
    """
    Calculates kalman-smoother estimates of SSM state posterior.
    :param X:       data, [d, t_max] numpy array
    :param y_init:  initial latent state, [k,] numpy array
    :param Q_init:  initial variance, [k, k] numpy array
    :param A:       latent dynamics matrix, [k, k] numpy array
    :param Q:       innovariations covariance matrix, [k, k] numpy array
    :param C:       output loading matrix, [d, k] numpy array
    :param R:       output noise matrix, [d, d] numpy array
    :param mode:    'forw' or 'filt' for forward filtering, 'smooth' for also backward filtering
    :return:
    y_hat:      posterior mean estimates, [k, t_max] numpy array
    V_hat:      posterior variances on y_t, [t_max, k, k] numpy array
    V_joint:    posterior covariances between y_{t+1}, y_t, [t_max, k, k] numpy array
    likelihood: conditional log-likelihoods log(p(x_t|x_{1:t-1})), [t_max,] numpy array
    """
    d, k = C.shape
    t_max = X.shape[1]

    # dimension checks
    assert np.all(X.shape == (d, t_max)), "Shape of X must be (%d, %d), %s provided" % (d, t_max, X.shape)
    assert np.all(y_init.shape == (k,)), "Shape of y_init must be (%d,), %s provided" % (k, y_init.shape)
    assert np.all(Q_init.shape == (k, k)), "Shape of Q_init must be (%d, %d), %s provided" % (k, k, Q_init.shape)
    assert np.all(A.shape == (k, k)), "Shape of A must be (%d, %d), %s provided" % (k, k, A.shape)
    assert np.all(Q.shape == (k, k)), "Shape of Q must be (%d, %d), %s provided" % (k, k, Q.shape)
    assert np.all(C.shape == (d, k)), "Shape of C must be (%d, %d), %s provided" % (d, k, C.shape)
    assert np.all(R.shape == (d, d)), "Shape of R must be (%d, %d), %s provided" % (d, k, R.shape)

    y_filt = np.zeros((k, t_max))  # filtering estimate: \hat(y)_t^t
    V_filt = np.zeros((t_max, k, k))  # filtering variance: \hat(V)_t^t
    y_hat = np.zeros((k, t_max))  # smoothing estimate: \hat(y)_t^T
    V_hat = np.zeros((t_max, k, k))  # smoothing variance: \hat(V)_t^T
    K = np.zeros((t_max, k, X.shape[0]))  # Kalman gain
    J = np.zeros((t_max, k, k))  # smoothing gain
    likelihood = np.zeros(t_max)  # conditional log-likelihood: p(x_t|x_{1:t-1})

    I_k = np.eye(k)

    # forward pass

    V_pred = Q_init
    y_pred = y_init

    for t in range(t_max):
        x_pred_err = X[:, t] - C.dot(y_pred)
        V_x_pred = C.dot(V_pred.dot(C.T)) + R
        V_x_pred_inv = np.linalg.pinv(V_x_pred)
        likelihood[t] = -0.5 * (np.linalg.slogdet(2 * np.pi * (V_x_pred))[1] +
                                x_pred_err.T.dot(V_x_pred_inv).dot(x_pred_err))

        K[t] = V_pred.dot(C.T).dot(V_x_pred_inv)

        y_filt[:, t] = y_pred + K[t].dot(x_pred_err)
        V_filt[t] = V_pred - K[t].dot(C).dot(V_pred)

        # symmetrise the variance to avoid numerical drift
        V_filt[t] = (V_filt[t] + V_filt[t].T) / 2.0

        y_pred = A.dot(y_filt[:, t])
        V_pred = A.dot(V_filt[t]).dot(A.T) + Q

    # backward pass

    if mode == 'filt' or mode == 'forw':
        # skip if filtering/forward pass only
        y_hat = y_filt
        V_hat = V_filt
        V_joint = None
    else:
        V_joint = np.zeros_like(V_filt)
        y_hat[:, -1] = y_filt[:, -1]
        V_hat[-1] = V_filt[-1]

        for t in range(t_max - 2, -1, -1):
            J[t] = V_filt[t].dot(A.T).dot(np.linalg.pinv(A.dot(V_filt[t]).dot(A.T) + Q))
            y_hat[:, t] = y_filt[:, t] + J[t].dot((y_hat[:, t + 1] - A.dot(y_filt[:, t])))
            V_hat[t] = V_filt[t] + J[t].dot(V_hat[t + 1] - A.dot(V_filt[t]).dot(A.T) - Q).dot(J[t].T)

        V_joint[-2] = (I_k - K[-1].dot(C)).dot(A).dot(V_filt[-2])

        for t in range(t_max - 3, -1, -1):
            V_joint[t] = V_filt[t + 1].dot(J[t].T) + J[t + 1].dot(V_joint[t + 1] - A.dot(V_filt[t + 1])).dot(J[t].T)

    return y_hat, V_hat, V_joint, likelihood

def logdet(A):
    t = np.linalg.cholesky(A)
    diag = np.diag(t)
    sum = np.sum(np.log(diag))*2
    
    return sum

def tran(x):
    ls = []
    for i in x:
        ls.append([i])
        
    ls = np.array(ls)
    return ls
        

def C_new(X,Y):
    d = X.shape[0]#1000
    k = Y.shape[0]#4
    t = X.shape[1]#5
    part1 = np.zeros((t,k))
    part2 = np.zeros((k,k))
    for i in range(d):
        part1+= tran(X[i]).dot(np.mat(Y[:,i]))
        part2 += tran(Y[:,i]).dot(np.mat(Y[:,i]))
        
    return part1 @ np.linalg.pinv(part2)
        
        
        

if __name__ =='__main__':
    X = np.loadtxt('ssm_spins.txt')
    y_init = np.random.multivariate_normal(np.zeros(4),np.identity(4))
    pi = math.pi
    A = 0.99*np.array([[math.cos(2*pi/180),-math.sin(2*pi/180),0,0],
                            [math.sin(2*pi/180),math.cos(2*pi/180),0,0],
                            [0,0,math.cos(2*pi/90),-math.sin(2*pi/90)],
                            [0,0,math.sin(2*pi/90),math.cos(2*pi/90)]])
    Q_init = np.identity(4)
    Q = np.identity(4)-A @ A.T
    C = np.array([[1,0,1,0],
                  [0,1,0,1],
                  [1,0,0,1],
                  [0,0,1,1],
                  [0.5,0.5,0.5,0.5]])
    R = np.identity(5)
    #AAAA = run_ssm_kalman(X.T, y_init, Q_init, A, Q, C, R, mode='smooth')
    #print(np.linalg.cholesky(AAAA[1][0,:,0].reshape((2,2))))
    
 
    Y,V,Vj,L = run_ssm_kalman(X.T, y_init, Q_init, A, Q, C, R, mode='filt')
    plt.plot(Y.T)
    ls = []
    '''
    for i in range(V.shape[0]):
        try:
            ls.append(logdet(V[i]))
        except:
            pass
    '''
    for i in range(V.shape[0]):
        ls.append(logdet(V[i]))
    plt.plot(ls)
    plt.show()
    
    
    