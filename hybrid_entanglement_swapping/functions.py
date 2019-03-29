# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:52:43 2019

@author: twink
"""
#libraries
import numpy as np

def probability_n_photon_pair(n, k_max, gamma, g, eta = 1):
    #print('n' ,n)
    lenghth_k = k_max+2;
    P_total =0;
    for i in range(lenghth_k):
        k = i-1; #when i runs from 0 to k_max+1, while k runs from -1 to k_max
        #print('k',k)
        P_total += P(k,n, gamma,g, eta)
    return P_total
        
def P(k,n, gamma, g, eta = 1):
    if n == k:
        P = f_a( gamma, k, g, eta)
        #print('a ', P)
    elif n == k + 2 :
        P = f_c( gamma, k, eta)
        #print('c' , P)
    else:
        P = 0
        
    return P

def T_00kk (gamma, k):
    T = ( 2*(gamma -1)**(k) ) / ( (gamma+1)**(k+1) ) ;
    return T

def T_11kk (gamma, k,g):
    # when g=tanh r, gamma =1. This function will give temp1 = inf . 
    # the plots will have some nan values
    temp1 = 2*(gamma -1)**(k-1)
    temp2 = (gamma+1)**(k+2) 
    temp3 = (gamma - 2*g**2 +1)*(gamma - 1) + 4*k*g**2
    T = temp1 / temp2 * temp3;
    return T

def T_10kplus1k(gamma, k, g):
    T = 4*g*np.sqrt(k+1) * ( (gamma-1)**k ) / ( (gamma + 1)**(k+2) );
    return T

def T_00kplus1kplus1(gamma, k):
    T= (2 * (gamma-1)**(k+1))/( (gamma+1)**(k+2) );
    return T

def f_a(gamma, k, g, eta = 1):
    if k == -1:
        a = 0;
    else:
        a =  (1-eta)*T_00kk(gamma, k ) + 0.5 * eta*  T_11kk(gamma, k , g);
    
    return a

def f_b( gamma, k, g, eta = 1):
    if k == -1:
        b = 0;
    else:
        b =  0.5 * eta*  T_10kplus1k(gamma, k , g);
    return b

def f_c( gamma, k, eta = 1):
    c = eta*0.5* T_00kplus1kplus1(gamma, k);
    return c