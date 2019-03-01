# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:04:16 2019

@author: twink
"""
import numpy as np
import turbulence_transmission_coefficient as turbulence
import matplotlib.pyplot as plt

import hybrid_entanglement_swapping as hybrid

"""
TESTS
"""
def test_DV_key_rate():
    f = 1.22
    q = 1/2
    e_0 = 1/2
    e_d = 0.015

    Y_0A =6.02e-6
    Y_0B = Y_0A
    detection_efficiency = 0.145
    
    transmissivity1 = np.logspace(-10, 0, 100);
    transmissivity2 = transmissivity1
    loss_dB = -10*np.log10(transmissivity1*transmissivity2);
    
    source = 'hybrid'
    r=1
    source_parameter = r
    K_hybrid = key_rate_DV(transmissivity1, transmissivity2, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
    
    source = 'PDC_II'
    mu = np.linspace(0.01,0.5,int(0.5/0.01));
    lambd = mu/2;
    source_parameter = lambd
    K_PDCII = key_rate_DV(transmissivity1, transmissivity2, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
    
    fig1, ax1 = plt.subplots()
    plt.plot(loss_dB, K_hybrid, label = 'hybrid')
    print(loss_dB.shape, K_PDCII.shape, transmissivity1.shape)
    plt.plot(loss_dB, K_PDCII, label = 'PDC type II')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
"""
FUNCTIONS
"""

def  mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a, num_points , \
                                  source, source_parameter, detection_efficiency,Y_0A, Y_0B, e_0,e_d, q, f):
    #initialize a few arrays
    mean_transmissivity = np.zeros( n_sigma_a) ;
    mean_key_rate = np.zeros(n_sigma_a) ;
    tc_all = np.zeros([n_sigma_a, num_points]);
    k_all = np.zeros([n_sigma_a, num_points]);

    for i in range(n_sigma_a):

        T1 = transmissivity_1[i,:]
        T2 = transmissivity_2[i,:]
        k = key_rate_DV(T1, T2, detection_efficiency,\
                    Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
        
        tc_all[i,:]= T1;
        k_all[i,:]=k;
        
        transmissivity = T1 * T2;   
        mean_transmissivity[i] = np.mean(transmissivity);
        mean_key_rate[i] = np.mean(k);


    mean_loss_dB = -10 * np.log10(mean_transmissivity);
    
    return tc_all, k_all, mean_loss_dB, mean_key_rate


def key_rate_DV(transmissivity1, transmissivity2, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter):
#    eta_B = transmissivity1 * detection_efficiency;
#    eta_A = transmissivity2 * detection_efficiency;
    
    if source == 'PDC_II':
        lambd = source_parameter
        Q, E = QE_lambda (lambd,e_0, e_d, Y_0A, Y_0B, transmissivity1, transmissivity2, detection_efficiency)
        K = Koashi_Preskill_keyrate(q,Q, E, f);
        K_max_lambd = np.max(K, axis = 0)
        
        #set negative key rate to 0
        filter_ = (K_max_lambd>0)
        K_max_lambd = K_max_lambd * filter_
        return K_max_lambd
    elif source == 'hybrid':
        r = source_parameter
        Q,E = QE_hybrid( r, e_0, e_d,  Y_0A, Y_0B, transmissivity1, transmissivity2, detection_efficiency);
        K = Koashi_Preskill_keyrate(q,Q, E, f);
        
        #set negative keyrate to 0
        filter_ = (K > 0)
        K = K * filter_
        return K
    else:
        raise ValueError('source is not recognized')
    
    
def Koashi_Preskill_keyrate(q,Q_lambda, E_lambda, f):
    R = q* Q_lambda * (1-(f+1)*binary_entropy(E_lambda)); 
    return R

def QE_hybrid(r, e_0, e_d,  Y_0A, Y_0B, transmissivity_A, transmissivity_B, detection_efficiency):
    eta_DV_entanglement = transmissivity_A*transmissivity_B
    loss  = 0#1 - transmissivity_A * transmissivity_B
    eta_A = detection_efficiency
    eta_B = detection_efficiency
    
    length_loss = len(eta_DV_entanglement) #len(loss)
    
    
    #find g_opt numerically. because using g = tanhr will results in inf in T_11kk from time to time
    # need to generalize this

    r = 2.395
    g = 0.9952380952380951

    V_an = 0.5 * ( (1-loss)* np.exp(2*r) + loss );
    V_sq = 0.5 * ( (1-loss)* np.exp(-2*r) + loss) ;
    tau = 0.5 * V_an *(1 - 1/g)**2  + 0.5 * V_sq *(1 + 1 / g)**2;
    gamma = g**2*(2*tau + 1) ; #gamma is now a function of loss
    
    k_max =100
    n_max = 50
    EQ = np.zeros(length_loss)
    Q = np.zeros(length_loss)
    for n in range(n_max+1):

        #P_n is a function of loss
        P_n = hybrid.probability_n_photon_pair(n, k_max, eta_DV_entanglement, gamma, g)
        Y_n= yield_n(n, eta_A, eta_B, Y_0A, Y_0B)
        e_n = error_rate(n, e_0, e_d, eta_A, eta_B, Y_n)
        Q += Y_n * P_n
        EQ +=  e_n * Y_n * P_n
    E = EQ/Q;
    return Q, E #function of loss
        
        
def QE_hybrid_no_loss(r, e_0, e_d,  Y_0A, Y_0B, eta_A, eta_B):
    #this function is deprecated
    g = np.tanh(r);
    P0 = (1-g**2)/2;
    P1 = (1+g**2)/2;
    #type cast P0 and P1 to array so that the transpose can work on both int and array
    P0 = np.array(P0); 
    P1 = np.array(P1)
    
    Y0 = yield_n(0, eta_A, eta_B, Y_0A, Y_0B);
    Y1 = yield_n(1, eta_A, eta_B, Y_0A, Y_0B);

    e_1 = e_0 - (e_0 - e_d) * eta_A* eta_B / Y1;
    
    Q = P0.T.dot(Y0)  +  P1.T.dot(Y1);
    EQ = P0.T.dot(e_0*Y0) + P1.T.dot(e_1*Y1) ;
    E = EQ/Q;
    return Q, E

def QE_lambda (lambd,e_0, e_d, Y_0A, Y_0B, transmissivity_A, transmissivity_B, detection_efficiency):
    eta_A = transmissivity_A * detection_efficiency
    eta_B = transmissivity_B * detection_efficiency
    #type cast lamd to array of array so that we can transpose it
    lambd = np.array([lambd])
    eta_A = np.array([eta_A])
    eta_B = np.array([eta_B])
    Q = Q_lambda(lambd, Y_0A, Y_0B, eta_A, eta_B);        
    EQ= E_lambda_times_Q_lambda(Q, lambd, e_0, e_d,  eta_A, eta_B);    
    E = EQ/ Q
    return Q, E

def Q_lambda(lambd, Y_0A, Y_0B, etaA, etaB):
    temp1 = (1 - Y_0A)/(1 +  lambd.T.dot(etaA))**2; # without .  , 1/a is the inverse matrix
    temp2 = (1 - Y_0B)/(1 + lambd.T.dot(etaB ) )**2;
    temp3 = (1-Y_0A)*(1-Y_0B)/(1 + lambd.T.dot(etaA) + lambd.T.dot(etaB) - lambd.T.dot(etaA*etaB))**2;
    Q =1 - temp1 - temp2 + temp3;
    return Q

def E_lambda_times_Q_lambda(Q_lambda,lambd, e_0, e_d,  etaA, etaB):
    temp1 = 2*(e_0-e_d)*(lambd.T + lambd.T**2).dot(etaA*etaB);
    temp2 = (1 + lambd.T.dot(etaA))*(1 + lambd.T.dot(etaB))*(1 + lambd.T.dot(etaA) + lambd.T.dot(etaB) - lambd.T.dot(etaA*etaB));
    EQ = e_0*Q_lambda - temp1/temp2;
    return EQ

def error_rate(n, e_0, e_d, eta_A, eta_B, Y_n):
    if n ==1:
        e_n = e_0 - (e_0 - e_d) * eta_A * eta_B / Y_n
    elif n == 2:
        e_n = 2*(e_0 - e_d )/ (3*Y_n) * (1-(1-eta_A)**2)*(1-(1-eta_B)**2)
    else: # for all larger n
        temp1 = 2*(e_0 - e_d)/(n+1)/Y_n
        temp2 = (1-(1-eta_A)**(n+1) * (1- eta_B)**(n+1) ) / ( 1-(1- eta_A)*(1 - eta_B) )
        if eta_A != eta_B:
            temp3 = ((1 - eta_A)**(n + 1) - (1- eta_B)**(n+1) )/ (eta_B - eta_A)
        else: # this is the limit of temp3 when eta_B tends to eta_A. Used l'hopital rule
            temp3 = (n+1) * (1-eta_A)**n     
        e_n = temp1 *( temp2 - temp3)
    return e_n

def yield_n(n, etaA, etaB, Y0A, Y0B):
    temp1 = 1-(1-Y0A)*(1-etaA)**n;
    temp2 = 1-(1-Y0B)*(1-etaB)**n;
    Y = temp1*temp2;
    return Y


def binary_entropy(x):
    H = - x* np.log2(x) - (1-x) * np.log2(1-x);
    return H






