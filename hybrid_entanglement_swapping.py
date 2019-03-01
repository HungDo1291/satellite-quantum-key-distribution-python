# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:21:47 2019

@author: twink
"""
import numpy as np
import matplotlib.pyplot as plt
#for surface plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


"""
TESTS
"""
def test_hybrid():

    r = np.linspace(0.1,2, int(2/0.1));
    length_r = len(r)
    k_max = 100
    
    # find E_LN for a matrix of r and g   
    g = np.linspace(0.1, 2, int(2/0.005));
    g_opt_PPT = np.tanh(r)
    
    #find E_LN in different conditions of loss and efficiency
    eta = 1
    loss = 0
    _,_,E_LN_1_0 = f_ELN( eta,loss, r, g, k_max, 2);  #dimension =2 when both r and g are free variables. E_LN is a 2D matrix
    g_opt_LN_1_0, E_max_LN_1_0 = g_tunning(E_LN_1_0, length_r, g);# find max of E_LN
    _,_,E_LN_PPT_1_0 = f_ELN( eta,loss, r, g_opt_PPT, k_max, 1); #dimension =1 when only r is a free variable.g = g_opt_PPT = tanh r. E_LN is a 1D matrix
    
    eta = 1
    loss = 0.2
    _,_,E_LN_1_0p2 = f_ELN( eta,loss, r, g, k_max, 2);  
    g_opt_LN_1_0p2, E_max_LN_1_0p2 = g_tunning(E_LN_1_0p2, length_r, g);
    _,_,E_LN_PPT_1_0p2 = f_ELN( eta,loss, r, g_opt_PPT, k_max, 1);  
    
    eta = 0.7
    loss = 0
    _,_,E_LN_0p7_0 = f_ELN( eta,loss, r, g, k_max, 2);  
    g_opt_LN_0p7_0, E_max_LN_0p7_0 = g_tunning(E_LN_0p7_0, length_r, g);
    _,_,E_LN_PPT_0p7_0 = f_ELN( eta,loss, r, g_opt_PPT, k_max, 1); 
    
    eta = 0.7
    loss = 0.2
    _,_,E_LN_0p7_0p2 = f_ELN( eta,loss, r, g, k_max, 2);  
    g_opt_LN_0p7_0p2, E_max_LN_0p7_0p2 = g_tunning(E_LN_0p7_0p2, length_r, g);
    _,_,E_LN_PPT_0p7_0p2 = f_ELN( eta,loss, r, g_opt_PPT, k_max, 1);  
        
#    takeda 15
    # NEED TO CHECK THIS AGAIN LATER
#    
#    g_opt_PPT = tanh(r)
#    
#    ket_10 = np.array([0, 0, 1, 0]).reshape(4,1)
#    ket_01 = np.array([0, 1, 0, 0]).reshape(4,1)
#    ket_00 = np.array([1, 0, 0, 0]).reshape(4,1)
#    
#    for i in range(length_r):
#        g_opt = g_opt_PPT[i];
#        psi_AD = (ket_10 + g_opt*ket_01) / np.sqrt(1+g_opt**2);
#        rho = (1+g_opt**2) / 2 * (psi_AD * (psi_AD.T) )+ (1-g_opt**2) / 2 *ket_00 * (ket_00.T);
#        rho_PT = PartialTranspose(rho);
#        E_LN[i] = np.log2( np.matrix.trace ( np.sqrt ( rho_PT * (rho_PT.T) )));
    
    # PLOTTING
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(g, r)
    surf = ax.plot_surface(X,Y,E_LN_1_0, cmap=cm.coolwarm,\
                       linewidth=0, antialiased=False)
    ax.set_xlabel('g')
    ax.set_ylabel('r')
    ax.set_zlabel('E_{LN}')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    plt.figure()
    plt.plot(r,E_max_LN_1_0,'k-', label = 'E^{max}_{LN} \eta =1.0, l=0.0')
    plt.plot(r,E_max_LN_1_0p2,'b-', label = 'E^{max}_{LN} \eta =1.0, l=0.2')
    plt.plot(r,E_max_LN_0p7_0,'g-', label = 'E^{max}_{LN} \eta =0.7, l=0.0')
    plt.plot(r,E_max_LN_0p7_0p2,'r-', label = 'E^{max}_{LN} \eta = 0.7, l=0.2')
    plt.plot(r,E_LN_PPT_1_0,'k--', label = 'E^{PPT}_{LN} \eta =1.0, l=0.0')
    plt.plot(r,E_LN_PPT_1_0p2,'b--', label = 'E^{PPT}_{LN} \eta =1.0, l=0.2')
    plt.plot(r,E_LN_PPT_0p7_0,'g--', label = 'E^{PPT}_{LN} \eta =0.7, l=0.0')
    plt.plot(r,E_LN_PPT_0p7_0p2,'r--', label = 'E^{PPT}_{LN} \eta = 0.7, l=0.2')
    #plt.plot(r,E_LN,'k:', label = 'E_{PPT}^{LN} in takeda15')
    
    plt.xlabel('r')
    plt.ylabel('Logarithmic negativity E_{LN}')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(r,g_opt_LN_1_0,'k')
    plt.plot(r,g_opt_LN_1_0p2,'b')
    plt.plot(r,g_opt_LN_0p7_0,'g')
    plt.plot(r,g_opt_LN_0p7_0p2,'r')
    plt.plot(r,g_opt_PPT,'k:')
    
    
#    legend('g^{opt}_{LN} when \eta =1.0, l=0.0',...
#        'g^{opt}_{LN} when \eta =1.0, l=0.2', ...
#        'g^{opt}_{LN} when \eta =0.7, l=0',...
#        'g^{opt}_{LN} when \eta =0.7, l=0.2',...
#        'g^{PPT}_{LN} = tanh r','Location','bestoutside')
    plt.title('g^{opt} ')
    plt.xlabel('r')
    plt.ylabel('g^{opt}')
    plt.show()
    
    return r, g, g_opt_LN_1_0p2, E_max_LN_1_0p2
    
def test_probability_n_photon_pair():
    #find E_LN in different conditions of loss and efficiency
    eta = 1
    l = 0.2
    k_max = 100
    
    #find g_opt numerically. because using g = tanhr will results in inf in T_11kk from time to time
    r = np.linspace(0.1,2.395,10)
    g = np.linspace(0.1, 2, int(2/0.005));
    _,_,E_LN_1_0p2 = f_ELN( eta,l, r, g, k_max, 2);  
    g_opt_LN_1_0p2, E_max_LN_1_0p2 = g_tunning(E_LN_1_0p2, len(r), g);
    
    print('r, g_opt, g=tanh r = ', r[9],g_opt_LN_1_0p2[9], np.tanh(r)[9] )

    r = r[9]
    g = g_opt_LN_1_0p2[9]

    V_an = 0.5 * ( (1-l)* np.exp(2*r) + l );
    V_sq = 0.5 * ( (1-l)* np.exp(-2*r) + l) ;
    tau = 0.5 * V_an *(1 - 1/g)**2  + 0.5 * V_sq *(1 + 1 / g)**2;
    gamma = g**2*(2*tau + 1) ;

    P_0 = probability_n_photon_pair(0, k_max, eta, gamma, g)
    P_1 = probability_n_photon_pair(1, k_max, eta, gamma, g) 
    P_2 = probability_n_photon_pair(2, k_max, eta, gamma, g)
    P_3 = probability_n_photon_pair(3, k_max, eta, gamma, g)
    P_4 = probability_n_photon_pair(4, k_max, eta, gamma, g)
    P_50 = probability_n_photon_pair(50, k_max, eta, gamma, g)
    print('P(0,1,2,3,4,50) = ', P_0, P_1, P_2, P_3, P_4, P_50)

'''
FUNCTIONS
'''
#def f_tau(l, r,  g):
#    len_r = len(r)
#    len_g = len(g)
#    V_an = 0.5 * ( (1-l)* np.exp(2*r) + l );
#    V_sq = 0.5 * ( (1-l)* np.exp(-2*r) + l) ;
#    t = np.zeros([len_r, len_g]);
#    for i in range(len_r):
#        for j in range(len_g):
#            t[i,j] = 0.5 * V_an[i] *(1-1/g[j])**2 + 0.5 * V_sq[i] *(1+1/g[j])**2;
#    
#    return t

def probability_n_photon_pair(n, k_max, eta, gamma, g):
    #print('n' ,n)
    lenghth_k = k_max+2;
    P_total =0;
    for i in range(lenghth_k):
        k = i-1; #when i runs from 0 to k_max+1, while k runs from -1 to k_max
        #print('k',k)
        P_total += P(k,n, eta, gamma,g)
    return P_total
        
def P(k,n, eta,gamma, g):
    if n == k:
        P = f_a(eta, gamma, k, g)
        #print('a ', P)
    elif n == k + 2 :
        P = f_c(eta, gamma, k)
        #print('c' , P)
    else:
        P = 0
        
    return P


def f_tau(l, r,  g):
    V_an = 0.5 * ( (1-l)* np.exp(2*r) + l );
    V_sq = 0.5 * ( (1-l)* np.exp(-2*r) + l) ;
    t = 0.5 * V_an *(1 - 1/g)**2 + 0.5 * V_sq *(1 + 1 / g)**2;
    return t

def f_gamma(g, tau):
    gamma = g**2*(2*tau + 1) ;
    return gamma

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

def f_a(eta, gamma, k, g):
    if k == -1:
        a = 0;
    else:
        a =  (1-eta)*T_00kk(gamma, k ) + 0.5 * eta*  T_11kk(gamma, k , g);
    
    return a

def f_b(eta, gamma, k, g):
    if k == -1:
        b = 0;
    else:
        b =  0.5 * eta*  T_10kplus1k(gamma, k , g);
    return b

def f_c(eta, gamma, k):
    c = eta*0.5* T_00kplus1kplus1(gamma, k);
    return c

def sum_negative_eigen_values(k_max, eta, gamma, g):
    lenghth_k = k_max+2;
    summation = 0#np.zeros(gamma.shape);

    for i in range(lenghth_k):
        k = i-1; #when i runs from 0 to k_max+1, while k runs from -1 to k_max
        a = f_a(eta, gamma, k, g);
        b = f_b(eta, gamma, k, g);
        c = f_c(eta, gamma, k);
        lambd = 0.5 * ( a+c - np.sqrt((a - c)**2 + 4 * b**2) );
        summation +=  ( np.abs(lambd) - lambd ) ;

    return summation


def f_ELN( eta, loss, r,g,k_max, dimension):
    length_r = len(r)
    length_g = len(g)
    
    if dimension == 2:
        tau = np.zeros([length_r, length_g]);
        gamma = np.zeros([length_r, length_g]);
        for i in range(length_r):
            for j in range(length_g):
                
                tau[i,j] = f_tau(loss, r[i], g[j]);
                gamma[i,j] = f_gamma(g[j], tau[i,j]);
    else: # if dimention !=2
        tau = np.zeros(length_r);
        gamma = np.zeros(length_r);
        for i in range(length_r):
                tau[i] = f_tau(loss, r[i], g[i]);
                gamma[i] = f_gamma(g[i],tau[i]);

    summation = sum_negative_eigen_values(k_max, eta, gamma, g);
    E_LN = np.log2(1+summation);
    return tau, gamma, E_LN


def g_tunning(E_LN, length_r, g):
    E_max_LN = np.zeros(length_r);
    g_opt_LN = np.zeros(length_r);
    for i in range( length_r) :

        index_max = np.argmax(E_LN[i,:]);
        E_max_LN[i] = E_LN[i,index_max]
        g_opt_LN[i] = g[index_max];
        if E_LN[i,index_max] != E_max_LN[i]:

            raise ValueError('the location of peak is not correct')

    return g_opt_LN, E_max_LN


if __name__ == "__main__":
    print(' TEST HYBRID PROTOCOL ')
    test_hybrid()
    print('TEST PROBABILITY n-PHOTON PAIRS')
    test_probability_n_photon_pair()
    