# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:04:16 2019

@author: twink
"""
import numpy as np
import turbulence_transmission_coefficient as turbulence

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
    eta_B = transmissivity1 * detection_efficiency;
    eta_A = transmissivity2 * detection_efficiency;
    
    if source == 'PDC_II':
        lambd = source_parameter
        Q, E = QE_lambda (lambd,e_0, e_d, Y_0A, Y_0B, eta_A, eta_B)
        K = Koashi_Preskill_keyrate(q,Q, E, f);
        K_max_lambd = np.max(K, axis = 0)
        
        #set negative key rate to 0
        filter_ = (K_max_lambd>0)
        K_max_lambd = K_max_lambd * filter_
        return K_max_lambd
    elif source == 'hybrid':
        r = source_parameter
        Q,E = QE_hybrid_no_loss(r, e_0, e_d,  Y_0A, Y_0B, eta_A, eta_B);
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

def QE_hybrid_no_loss(r, e_0, e_d,  Y_0A, Y_0B, eta_A, eta_B):
    
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

def QE_lambda (lambd,e_0, e_d, Y_0A, Y_0B, eta_A, eta_B):
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

def yield_n(n, etaA, etaB, Y0A, Y0B):
    temp1 = 1-(1-Y0A)*(1-etaA)**n;
    temp2 = 1-(1-Y0B)*(1-etaB)**n;
    Y = temp1*temp2;
    return Y


def binary_entropy(x):
    H = - x* np.log2(x) - (1-x) * np.log2(1-x);
    return H






