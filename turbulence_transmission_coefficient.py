# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:35:05 2019

@author: Hung Do
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv #Modified Bessel function of the first kind of real order.
#################################
#CIRCULAR FADING CHANNEL #############################

#######################################
def circular_beam_wandering_trans_coef(sigma_a, W0, L, wavelength,aperture_radius,  num_data_points, num_simulation_points):
    # generate the function of t_approx by r_a, which only depends on W_a

    #random.seed(1) #initialize the pseudo-random
    z_R = np.pi * W0 **2 / wavelength
    W_a = W0*np.sqrt(1+L**2/z_R**2)/aperture_radius;
    
    dr_a = 4*sigma_a/num_data_points;
    r_a = np.linspace(dr_a, 4*sigma_a, 4*sigma_a/dr_a)

    t_approx= np.sqrt(circular_beam_approximated_transmittance(r_a, W_a));
    
    #simulate r_a according to beam wandering distribution with variance
    x = np.random.normal(0,sigma_a, num_simulation_points);
    y = np.random.normal(0,sigma_a, num_simulation_points);

    r_a_simulated = np.sqrt(x**2 + y**2);

    # find t from r_a_simulated
    t_approx_simulated = np.zeros(num_simulation_points);
    for i in range(num_simulation_points):
        index = np.argmin(np.abs(r_a - r_a_simulated[i]));
        t_approx_simulated[i]=t_approx[index];


    
    if 1: # set to 1 to enable test plotting
        #plotting
        fig1 = plt.figure(1)
        title_text = 'Simulation of ' + str(num_simulation_points) + 'rand points, given d=0, W = '+str(W_a)+', sigma = '+str(sigma_a)+' a, numerical integration with '+str(num_data_points)+' data points'
        
        plt.hist( x, bins = 100, histtype = 'step', density = True, label = 'x_a')
        plt.hist( y, bins = 100, histtype = 'step', density = True, label = 'x_b')
        plt.hist(r_a_simulated, bins = 100, histtype = 'step', density = True, label = 'r_a simulated')

        plt.title(title_text)
        plt.legend()
        plt.grid(True)
        fig1.show()

        fig2 = plt.figure(2)
        plt.hist(r_a,bins = 50, histtype = 'step', density = True, label = 'r_a')
        plt.title (str(num_data_points) + 'data points')
        fig2.show()

        fig3, ax3 = plt.subplots()
        plt.hist(t_approx_simulated, bins = 100, density = True)
        #plot the PDTC from the pdf function in theory
        dT = 1/(num_data_points);
        T = np.linspace(dT, 1, 1/dT)
        plt.plot(T,PDTC(T,W_a,sigma_a,0),'b','DisplayName','W = 0.7a, sigma = 0.7a, d = 0')
        ax3.set_yscale('log')
        plt.legend()
        plt.title(title_text)
        plt.xlabel('t')
        plt.ylabel('probability')
        fig3.show()

    return r_a_simulated, t_approx_simulated 

def circular_approximation_parameters(W_a):
    t_0_squared = 1 - np.exp(-2 / W_a**2);
    temporary_var = 1 - np.exp(-4 / W_a**2)* iv(0,4/W_a**2);
    lambd = 8/W_a**2 * (np.exp(-4/W_a**2) * iv(1, 4/W_a**2) / temporary_var) * (1/(np.log(2 * t_0_squared / temporary_var)));
    R_a = (np.log(2*t_0_squared/temporary_var))**(-1/lambd);
    return t_0_squared, lambd, R_a

def circular_beam_approximated_transmittance(r_a, W_a): 
    [t_0_squared, lambd, R_a] = circular_approximation_parameters(W_a);
    y = (t_0_squared * np.exp(-(r_a/R_a)**lambd));
    return y

def PDTC(t, W_a,sigma_a, d_a):

    t_0_squared, lambd, R_a = circular_approximation_parameters(W_a)

    t_0 = np.sqrt(t_0_squared)
    temporary_var = 2 * np.log(t_0/t);
    temp1 = 2 * R_a**2 / (sigma_a**2 * lambd * t);
    temp2 = temporary_var ** (2 / lambd - 1);
    temp3 = iv(0, R_a*d_a/sigma_a**2 * temporary_var**(1/lambd));
    temp4 = np.exp(-1/(2*sigma_a**2) * (R_a**2 * temporary_var**(2/lambd) + d_a**2));
    y = temp1 * temp2 * temp3 * temp4;
    y = y * (t < t_0);
    return y
    
##############################
#   ELLIPTIC FADING CHANNEL
#############################

