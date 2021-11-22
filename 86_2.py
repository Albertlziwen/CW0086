# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:50:25 2021

@author: Albert
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import math
from scipy.special import beta,gamma
def main():
    # load the data set
    Y = np.loadtxt('C:\\Users\\Albert\\Desktop\\binary.txt')
    N, D = Y.shape

    # this is how you display one image using matplotlib,
    # e.g. the 4th image:
    y = Y[3,  :]
    plt.figure()
    plt.imshow(np.reshape(y, (8,8)),
               interpolation="None",
               cmap='gray')
    plt.axis('off')

    # now we will display the whole data set:
    plt.figure(figsize=(5, 5))
    for n in range(N):
        plt.subplot(10, 10, n+1)
        plt.imshow(np.reshape(Y[n, :], (8,8)),
                   interpolation="None",
                   cmap='gray')
        plt.axis('off')
    plt.show()
'''
if __name__ == "__main__":
    main()
'''
#1.d
def ML():
  Y = np.loadtxt('C:\\Users\\Albert\\Desktop\\binary.txt')
  P = []
  N, D = Y.shape
  t = np.sum(Y,axis = 0)
  for i in range(D):
    P.append(t[i]/N)
    
  return P
'''
  P = np.array(P)
  plt.figure(np.reshape(P,(8,8)),
             interpolation = 'None',
             cmap = 'gray')
  plt.axis('off')
  plt.show()
'''
  
#P1= ML()
#1.e
def MAP():
  Y = np.loadtxt('C:\\Users\\Albert\\Desktop\\binary.txt')
  P = []
  N, D = Y.shape
  t = np.sum(Y,axis = 0)
  alpha = 3
  beta = 3
  for i in range(D):
    P.append((t[i]+alpha-1)/(N+alpha+beta-2))
    
  return P

#P2 = MAP()
def log_gamma(n):
  lg = 0
  for i in range(1,n):
    lg+=np.log(i)
      
  return lg


#2.b unknown but identical p_d
def uip():
  Y = np.loadtxt('C:\\Users\\Albert\\Desktop\\binary.txt')
  N, D = Y.shape
  #the nimber of black pixels (x=1)
  t = np.sum(Y)
  #P = math.gamma(t+1)*math.gamma(D*N-t+1)/math.gamma(D*N+2)
  #P = beta(t+1,D*N-t+1)
  t = int(t)
  log_P = log_gamma(t+1)+log_gamma(D*N-t+1)-log_gamma(D*N+2)
  return log_P

#c seperate, unknown p_d
def sup():
  Y = np.loadtxt('C:\\Users\\Albert\\Desktop\\binary.txt')
  N, D = Y.shape
  log_P = 0
  for i in range(D):
    #\sum_{n=1}^N X_d^{n}
    t = np.sum(Y[:,i])
    t = int(t)
    log_P += log_gamma(t+1)+log_gamma(N+1-t)-log_gamma(N+2)
    
  
  return log_P


  

  
  







    