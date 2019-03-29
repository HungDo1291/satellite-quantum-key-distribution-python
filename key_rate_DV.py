# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:04:16 2019

@author: twink
"""
import numpy as np
import turbulence_transmission_coefficient as turbulence
import matplotlib.pyplot as plt

import hybrid_entanglement_swapping.imperfect_source as hybrid

"""
TESTS
"""

def test_zero_transmissivity():
    q = 1/2
    e_0 = 1/2
    e_d = 0.015

    Y_0A =6.02e-6
    Y_0B = Y_0A
    detection_efficiency = 0.145
    f = 1.22
    transmissivity_B = [0, 1e-5]
    transmissivity_A = np.ones(len(transmissivity_B))
    loss_dB = -10*np.log10(transmissivity_A*transmissivity_B);
    
    source = 'hybrid'
    loss_on = 'loss on DV entanglement'
    
    Q_hybrid_dv, E_hybrid_dv, k_hybrid_dv, P_n_dv = key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, 'hybrid', loss_on)
    
    print('Q, E, K, P(n)', Q_hybrid_dv, E_hybrid_dv, k_hybrid_dv, P_n_dv)
    
def test_DV_key_rate(f=1.22):

    q = 1/2
    e_0 = 1/2
    e_d = 0.015

    Y_0A =6.02e-6
    Y_0B = Y_0A
    detection_efficiency = 0.145
    
    transmissivity_B = np.linspace(0.9, 1, num=10000)
    transmissivity_A = np.ones(len(transmissivity_B))
    loss_dB = -10*np.log10(transmissivity_A*transmissivity_B);
    
    source = 'hybrid'
    r=1
    source_parameter = r
    loss_on = 'loss on CV entanglement'
    Q_hybrid_cv, E_hybrid_cv, k_hybrid_cv, P_n_cv = key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, 'hybrid', loss_on)
    loss_dB = -10 * np.log10(transmissivity_A * transmissivity_B)
    
    source = 'PDC_II'
    mu = np.linspace(0.01,0.5,int(0.5/0.01));
    lambd = mu/2;
    source_parameter = lambd
    Q_optimized, E_optimized, K_PDCII = key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
    
    fig1, ax1 = plt.subplots()
    plt.plot(loss_dB, k_hybrid_cv, label = 'hybrid')
    #print(loss_dB.shape, K_PDCII.shape, transmissivity_A.shape)
    plt.plot(loss_dB, K_PDCII, label = 'PDC type II')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()
    
    
    fig1, ax1 = plt.subplots()
    plt.plot(loss_dB, Q_hybrid_cv, label = 'Q')
    plt.plot(loss_dB, E_hybrid_cv, label = 'E')
    plt.plot(loss_dB, binary_entropy(E_hybrid_cv), label = 'H(E)')
    plt.grid()
    plt.xlabel('Optical link loss, dB')
    plt.legend()
    plt.show()
    
def test_H_K():
    E = 0.02159315
    Q = 0.01991333
    H = binary_entropy(E)
    print('H = ' ,  H)
    q = 0.5
    f = 1.22
    K = Koashi_Preskill_keyrate(q,Q, E, f)
    print('K = ' ,  K)
    
def test_QE_hybrid(loss_on = 'loss on CV entanglement'):
 
    e_0 = 1/2
    e_d = 0.015
    detection_efficiency = 0.145
    Y_0A = 6.02e-6
    Y_0B = Y_0A

    transmissivity_B=[0.938832883288]#np.linspace(0.93, 0.94,10000)
    transmissivity_A = np.ones(len(transmissivity_B))
    Q,E,P = QE_hybrid(loss_on, e_0, e_d,  Y_0A, Y_0B, transmissivity_A, transmissivity_B,\
              detection_efficiency)
    print("Q = ", Q)
    print("E = ", E)
    print("P = ", P)
    H = binary_entropy(E)
    print('H = ' ,  H)
    q = 0.5
    f = 1
    K = Koashi_Preskill_keyrate(q,Q, E, f)
    print('K = ', K)
    T_critical = transmissivity_B[np.argmax(K>0)]
    print('T_critical',T_critical)
    print('loss_critical in dB',-10*np.log10(T_critical))
            
    
def test_yield_error():
    n = 0;
    etaA = 0.145
    etaB = 0.145
    Y0A = 6.02e-6
    Y0B = Y0A
    Y0 = yield_n(n, etaA, etaB, Y0A, Y0B)
    Y0_expected = 3.6e-11
    tolerance = 1e-12
    if (np.abs(Y0-Y0_expected) > tolerance ):
        raise ValueError('Y0 = ' + str(Y0)+' instead of the expected value '+str(Y0_expected))
    else:
        print("Y(0) = Y_0A * Y_0B ", Y0, " as expected.")
    
    e_0 = 0.5
    e_d = 0.015 
    e0 = error_rate(n, e_0, e_d, etaA, etaB, Y0)
    print('e0 =', e0)
    
    e = np.zeros(10)
    Y = np.zeros(10)
    for n in range(10):
        Y[n] = yield_n(n, etaA, etaB, Y0A, Y0B)
        e[n] = error_rate(n, e_0, e_d, etaA, etaB, Y[n])
        
    print(Y, e)
    
    e1_approx = (e_0*(Y0A*Y0B + Y0A*etaB + etaA*Y0B) + e_d*etaA*etaB)/Y[1]
    
    fig, ax1 = plt.subplots()
    ax1.plot(Y, label ='yield Y(n)')
    ax1.plot(e, label ='error-rate e(n)')
    ax1.plot([1], [e1_approx],'ro')
    plt.xlabel('Photon-number n')
    plt.grid()
    plt.legend()
    plt.savefig('figures/yield_error.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    fig.show()
    
    #P when loss on DV 40dB
    P = np.array([9.95813267e-001, 4.16920387e-003, 1.74553418e-005, 7.30808485e-008,\
         3.05969971e-010, 1.28101445e-012, 5.36326498e-015, 2.24545563e-017, \
         9.40112230e-020, 3.93599852e-022])
    Q = np.sum(P* Y)
    EQ = sum(e*P*Y)
    E = EQ/Q
    print('Q, E = ', Q, E)
    
    q = 0.5
    f = 1.22
    K = Koashi_Preskill_keyrate(q,Q, E, f)
    print('K = ', K)
    
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
        Q, E, lambd = QE_lambda (lambd,e_0, e_d, Y_0A, Y_0B, transmissivity1, transmissivity2, detection_efficiency)
        K = Koashi_Preskill_keyrate(q,Q, E, f);
        K_max_lambd = np.max(K, axis = 0)
        #find Q,E
        n = len(transmissivity1)
        arg_max = np.argmax(K, axis =0)
        Q_optimized = Q[arg_max, np.arange(n)]
        E_optimized = E[arg_max, np.arange(n)]
        #lambd_average = np.average(lambd_for_K_max)
#        lambd_average = 0.1
#        n = np.array(np.arange(51))
#        P_n = (n+1)*lambd_average**n / (1+lambd_average)**(n+2)
        #set negative key rate to 0
        filter_ = (K_max_lambd>0)
        K_max_lambd = K_max_lambd * filter_
        return Q_optimized, E_optimized, K_max_lambd#, P_n
    elif source == 'hybrid':
        loss_on = source_parameter
        Q,E, P_n = QE_hybrid( loss_on, e_0, e_d,  Y_0A, Y_0B, transmissivity1, transmissivity2, detection_efficiency);
        K = Koashi_Preskill_keyrate(q,Q, E, f);
        
        #set negative keyrate to 0
        filter_ = (K > 0)
        K = K * filter_
        return Q,E, K, P_n
    else:
        raise ValueError("source should be either 'hybrid' or 'PDC_II'.")
    
    
def Koashi_Preskill_keyrate(q,Q_lambda, E_lambda, f):
    R = q* Q_lambda * (1-(f+1)*binary_entropy(E_lambda)); 
    return R

def QE_hybrid(loss_on, e_0, e_d,  Y_0A, Y_0B, transmissivity_A, transmissivity_B, detection_efficiency):
    
    length_loss = len(transmissivity_B) 
    if loss_on == 'loss on DV entanglement':
        eta_DV_entanglement = transmissivity_A*transmissivity_B
        loss_CV_entanglement  = 0
    elif loss_on == 'loss on CV entanglement':
        eta_DV_entanglement = 1
        loss_CV_entanglement  = 1 - transmissivity_A * transmissivity_B
    else:
        raise ValueError ("loss_on should be either 'loss on DV entanglement' or 'loss on CV entanglement'.")
        
    # loss at the DV detectors due to detection efficiency
    eta_A = detection_efficiency
    eta_B = detection_efficiency
      
    #find g_opt numerically. because using g = tanhr will results in inf in T_11kk from time to time
    # need to generalize this

    r = 2.395
    g = 0.9952380952380951

    V_an = 0.5 * ( (1-loss_CV_entanglement)* np.exp(2*r) + loss_CV_entanglement );
    V_sq = 0.5 * ( (1-loss_CV_entanglement)* np.exp(-2*r) + loss_CV_entanglement) ;
    tau = 0.5 * V_an *(1 - 1/g)**2  + 0.5 * V_sq *(1 + 1 / g)**2;
    gamma = g**2*(2*tau + 1) ; #gamma is now a function of loss on DV entanglement
    
    k_max =100
    n_max = 50
    EQ = np.zeros(length_loss)
    Q = np.zeros(length_loss)
    # record P(n)
    P_n = np.zeros([n_max+1, length_loss])
    for n in range(n_max+1):
        #P_n is a function of loss
        P_n[n, :] = hybrid.probability_n_photon_pair(n, k_max, eta_DV_entanglement, gamma, g)
        Y_n= yield_n(n, eta_A, eta_B, Y_0A, Y_0B)
        e_n = error_rate(n, e_0, e_d, eta_A, eta_B, Y_n)
        Q += Y_n * P_n[n, :]
        EQ +=  e_n * Y_n * P_n[n, :]
    E = EQ/Q;
    return Q, E, P_n #function of loss 
        
        
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
    return Q, E, lambd

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
    if n ==0:
        e_n = e_0
    elif n ==1:
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


if __name__ == "__main__":
    #print(' TEST yield and error ')
    #test_yield_error();
#    print('TEST QE_hybrid , loss on DV')
#    test_QE_hybrid('loss on DV entanglement')
#    print('TEST QE_hybrid , loss on CV')
#    test_QE_hybrid('loss on CV entanglement')
#    print('test_DV_key_rate():')
#    test_DV_key_rate(f=1)
    #print('test_H_K():')
    #test_H_K()
    print('test_zero_transmissivity():')
    test_zero_transmissivity()




