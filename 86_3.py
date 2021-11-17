import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def EM(K,X,iter):
    N,D = X.shape
    pi, P, R = initialize(N,K,D)
    likeli = []
    for t in range(iter):
        for k in range(K):
            for d in range(D):
                P[k,d] = R[:,k].dot(X[:,d])/np.sum(R[:,k])
                
        for k in range(K):
            pi[k] = np.sum(R[:,k])/N
        
        #update r_{nk}
        for n in range(N):
            for k in range(K):
                R[n][k] = pi[k]
                for d in range(D):
                    R[n][k] *= (P[k][d]**X[n][d])*((1-P[k][d])**(1-X[n][d]))
                    
        for i in range(N):
            R[i] = R[i]/np.sum(R[i])

        likeli.append(loglikelihood(X,P,R))



    return likeli
#we need initialize \pi_k and r_{nk}, using uniform distribution
def initialize(N,K,D):
    pi = np.ones(K)
    pi = pi/len(pi)
    P = np.ones((K,D))*0.27
    R = np.ones((N,K))*(1/len(pi))*(0.5**D)
    #normalization R, makes \sum_{k} r_{nk} = 1
    R = np.random.uniform(0,1,size = (N,K))
    for i in range(K):
        R[:i] = R[:i]/np.sum(R[:i])
    
    
    return pi, P, R

def loglikelihood(X,P,R):
    N,D = X.shape
    K = P.shape[0]
    likeli = 0
    for n in range(N):
        for d in range(D):
            temp = 0 
            for k in range(K):
                temp+= R[n][k]*(P[k,d]**X[n,d])*(1-P[k,d])**(1-X[n,d])
            likeli += np.log(temp)

    return likeli

if __name__ == '__main__':
    X = np.loadtxt('binarydigits.txt')
    K =2
    N,D = X.shape
    pi, P, R = initialize(N,K,D)
    likeli = EM(K,X,20)
    print('hhhhhh')
    plt.plot(likeli)
    plt.show()
                
