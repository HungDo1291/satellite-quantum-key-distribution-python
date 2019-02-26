# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:28:41 2019

@author: z5185265
"""
import numpy as np
import turbulence_transmission_coefficient as turbulence

#####################################
# CV CM AND KEY RATE
#######################################

def  mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a,   num_points, protocol, v, chi,mesurement_method, reconciliation_ref):
    #initialize a few arrays
    mean_transmissivity = np.zeros( n_sigma_a) ;
    mean_key_rate = np.zeros(n_sigma_a) ;
    tc_all = np.zeros([n_sigma_a, num_points]);
    k_all = np.zeros([n_sigma_a, num_points]);

    for i in range(n_sigma_a):

        T1 = transmissivity_1[i,:]
        T2 = transmissivity_2[i,:]
        transmissivity = T1*T2;   

        a, b, c = covariance_matrix(T1, T2, v, chi, protocol);
        k = key_rate_CV( a, b, c, mesurement_method, reconciliation_ref);
        
        mean_transmissivity[i] = np.mean(transmissivity);
        mean_key_rate[i] = np.mean(k);
        tc_all[i,:]= T1;
        k_all[i,:]=k;

    mean_loss_dB = -10 * np.log10(mean_transmissivity);
    
    return tc_all, k_all, mean_loss_dB, mean_key_rate

def key_rate_CV(a, b, c, method, reference):
	# calculate the sympletic eigenvalues
    z = np.sqrt((a+b)**2 - 4*c**2);
    nu1 = 0.5*(z + b - a);
    nu2 = 0.5*(z - b + a);
    entropy_e = f(nu1) + f(nu2);
    
    if reference == "A":
        if method == "homodyne":
            nu = np.sqrt(b*(b-c**2/a));
        
        elif method ==  "heterodyne":
            nu = b - c**2/(a+1);
        
    
    if reference == "B":
        if method == "homodyne":
            nu = np.sqrt(a*(a-c**2/b));
        
        elif method ==  "heterodyne":
            nu = a - c**2/(b+1);

    entropy_e_conditioned = f(nu);

    xi =1 ; #perfect reconcilation
    if method == "homodyne":
        I_ab = 0.5 * np.log2(a/(a-c**2/b));
    elif method ==  "heterodyne":
        I_ab = np.log2((b+1)/(b+1-c**2/(a+1)));
    else:
        print("invalid detection method");
	
    I_e = entropy_e - entropy_e_conditioned;
    key_rate = xi*I_ab - I_e;
    for i , value in enumerate(key_rate):
        if value <0:
            key_rate[i] = 0
            
    #set negative keyrate to zero
    filter_ = (key_rate>0)
    key_rate = key_rate*filter_;
    return key_rate

def covariance_matrix(T_a, T_b, v, chi, protocol):
    chi_a = chi;
    chi_b = chi;
    if protocol == 'satellite based entanglement':
    	a = 1 + T_a * (v-1) + chi
    	b = 1 + T_b*(v-1) +chi
    	c = np.sqrt(T_a *T_b)* np.sqrt(v**2 -1)
    elif protocol == 'entanglement swapping':
        theta = (v-1)*(T_a + T_b)+(chi_a + chi_b) + 2;
        a = v - (v**2-1)*T_a/theta;
        b = v - (v**2-1)*T_b/theta;
        c = (v**2-1)*np.sqrt(T_a*T_b)/theta;        
    elif protocol == 'direct transmission':
        a = v;
        b = 1 + T_a * T_b * (v-1) + chi;
        c = np.sqrt( T_a * T_b ) * np.sqrt( v**2 - 1 );
    else:
        raise ValueError('Protocol not recognized')
    
    return a, b, c


def f(x):
	y = (x+1)/2* np.log2((x+1)/2) - (x-1)/2 * np.log2((x-1)/2);
	n = len(x)
	for i in range(n):
		if x[i] == 1 :
			y[i] = 0;
			print("warning, taking log of 0 in calculating f(x)")
	return y
