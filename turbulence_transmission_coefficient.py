# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:35:05 2019

@author: Hung Do
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from scipy.special import iv #Modified Bessel function of the first kind of real order.
from scipy.special import lambertw #lambert_W function

"""
TEST FUNCTIONS: still need some corrections to match the new function for generating transmissivity
"""

def test_turbulence():
    
    sigma_a = np.linspace(0.01 ,0.51, 7)
    n_sigma = len(sigma_a)
    chi = 0.02 #excess noise at the receiver
    ref = 'B' #reversed reconciliation
    L= 500000 #distance from satellite to Earth : 500 km
    wavelength = 780*1e-9
    W0 = 0.12
    aperture_radius = 1
    num_simulation_points = int(1e5); #typecast to integer inorder to avoid errors
      
    _, tc1 = circular_beam_wandering_trans_coef(sigma_a[4], W0 , L, \
        wavelength, aperture_radius,  num_simulation_points, num_simulation_points, test = True)
    _, tc2 = elliptic_model_trans_coef(sigma_a[4], W0 , L, \
        wavelength, aperture_radius,  num_simulation_points, test = True)
    
    fig, ax = plt.subplots()
    for i in range(n_sigma):
        _, tc2 = elliptic_model_trans_coef(sigma_a[i], W0 , L, wavelength, aperture_radius,  num_simulation_points)    
        plt.hist(tc2**2, bins = 100, histtype = 'step', density = True, label = '\sigma = '+str(sigma_a[i]))
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Transmittance T=t^2')
    plt.yscale('log')
    plt.title('Beam-wandering model')
    #plt.ylim((1/num_simulation_points, 1e2))
    fig.show()
    
    fig2, ax2 = plt.subplots()
    for i in range(n_sigma):
        _, tc1 = turbulence.circular_beam_wandering_trans_coef(sigma_a[i], W0 , L, wavelength, aperture_radius, num_simulation_points, num_simulation_points)    
        plt.hist(tc1**2, bins = 100, histtype = 'step', density = True, label = '\sigma = '+str(sigma_a[i]))
    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Transmittance T=t^2')
    plt.yscale('log')
    plt.title('Elliptical model')
    #plt.ylim((1/num_simulation_points, 1e2))
    fig2.show()

#################################
#CIRCULAR FADING CHANNEL #############################

#######################################
def generate_transmissivity(sigma_a, turbulence_model, W0 , L, wavelength, aperture_radius,  num_points ):
    n_sigma_a = len(sigma_a)
    transmissivity_1 = np.zeros([n_sigma_a, num_points])
    transmissivity_2 = np.zeros([n_sigma_a, num_points])
    for i in range(n_sigma_a):
        print('sigma = ', sigma_a[i])
        if turbulence_model == 'elliptic':
            _, tc1 = elliptic_model_trans_coef( sigma_a[i], W0 , L, wavelength, aperture_radius,  num_points);
            _, tc2 = elliptic_model_trans_coef( sigma_a[i], W0 , L, wavelength, aperture_radius,  num_points);
        elif turbulence_model == 'circular':
            _, tc1 = circular_beam_wandering_trans_coef(sigma_a[i], W0 , L, wavelength, aperture_radius, num_points, num_points);
            _, tc2 = circular_beam_wandering_trans_coef(sigma_a[i], W0 , L, wavelength, aperture_radius, num_points, num_points);
        else:
            print('invalid turbulence_model')

        transmissivity_1[i,:] = tc1**2;
        transmissivity_2[i,:] = tc2**2;
        
    transmissivity_1.tofile('data/transmissivity_1_' + turbulence_model + '.dat')
    transmissivity_2.tofile('data/transmissivity_2_' + turbulence_model + '.dat')

    return transmissivity_1, transmissivity_2


def circular_beam_wandering_trans_coef(sigma_a, W0, L, wavelength,aperture_radius,  num_data_points, num_simulation_points, test = False):
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
    
    if test == True: # set to 1 to enable test plotting
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
        plt.plot(T,PDTC(T,W_a,sigma_a,0), label = 'PDTC')
        ax3.set_yscale('log')
        plt.legend()
        plt.title(title_text)
        plt.xlabel('t')
        plt.ylabel('probability')
        plt.ylim((1/num_simulation_points, 1e2))
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
    
###########################################
#   ELLIPTIC FADING CHANNEL
#   Author: Mingjian wrote in MATLAB
#   Translator: Hung translated to Python    
#############################################

def elliptic_model_trans_coef(sigma, W0, L, wavelength,a,num_simulation_points, test = False): 
    

    Omega = np.pi * W0**2 / L / wavelength;
    SigL_squared = sigma**2/ (0.33 * W0**2 * Omega**(-7/6));
    
    #Moments from ref [30]
    Var_x = 0.33 * W0**2 * SigL_squared * Omega**(-7/6);
    Var_A = np.log(1 + (1.2 * SigL_squared * Omega**(5/6)) / (1 + 2.96 * SigL_squared * Omega**(5/6))**2);
    Cov_A = np.log(1 - (0.8 * SigL_squared * Omega**(5/6)) / (1 + 2.96 * SigL_squared * Omega**(5/6))**2);

    Mea_A = np.log((1 + 2.96 * SigL_squared * Omega**(5/6))**2 \
    /(Omega**2 * np.sqrt((1 + 2.96 * SigL_squared * Omega**(5/6))**2 + 1.2 * SigL_squared * Omega**(5/6))));

    mu = [0, 0, Mea_A, Mea_A];
    SIGMA = [[Var_x, 0, 0, 0], [0, Var_x, 0, 0], [0, 0, Var_A, Cov_A], [0, 0, Cov_A, Var_A]];

    # Generate random varialbles
    x, y, w1, w2 = np.random.multivariate_normal(mu,SIGMA,num_simulation_points).T;

    W1 = W0 * np.sqrt(np.exp(w1));
    W2 = W0 * np.sqrt(np.exp(w2));
    
    temp = np.linspace(1, num_simulation_points, num_simulation_points)
    ang = temp * (np.pi/2)/num_simulation_points;
    ang = ang.T;
    ang0 = np.arctan(y / x);

    # Calculating using equations
    WeffSqInv = lambertw(4 * (a**2) / W1 / W2 \
            * np.exp(a**2 / W1**2 * (1 + 2 * np.cos(ang - ang0)**2)) \
            * np.exp(a**2 / W2**2 * (1 + 2 * np.sin(ang - ang0)**2)));

    I0 = iv(0, WeffSqInv);
    I1 = iv(1, WeffSqInv);

    Ri = np.log(2 * (1 - np.exp(-0.5 * WeffSqInv)) / (1 - np.exp(-WeffSqInv) * I0) );
    La = 2 * WeffSqInv *(np.exp(-WeffSqInv) * I1) / (1 - np.exp(-WeffSqInv) * I0) * Ri **(-1);
    R = Ri ** (-1 / La);

    LaFuc, RFuc = LaRFuc(1/W1 - 1/W2, a);
    T0 = 1 - iv(0, a**2 * (W1**2 - W2**2) / (W1**2 * W2**2)) \
    * np.exp(-a**2 * (W1**2 + W2**2) / (W1**2 * W2**2)) \
    - 2 * (1 - np.exp(-0.5 * a**2 * (1/W1 - 1/W2)**2)) \
    * np.exp(-((W1 + W2)**2 / np.abs(W1**2 - W2**2) / RFuc)**LaFuc);
    
    r0 = np.sqrt(x**2 + y**2);
    T = T0 * np.exp(-( r0 / a / R)**La);
    tc = np.sqrt(np.real(T));
    
    if test == True: #set to 1 to test this function by plotting
        fig, ax = plt.subplots()
        plt.hist(tc, bins = 100, histtype = 'step', density = True, label = 'sigma = '+str(sigma))
        plt.legend()
        plt.ylim((1/num_simulation_points, 1e2))
        ax.set_yscale('log')
        fig.show()
    
    return r0, tc


def LaRFuc( In, a ):
    #fix the divide by 0 problem
    for i, value in enumerate(In):
        if value ==0:
            print('In[',i,'] = ',value)
            In[i] = 1e-12
        if value == np.nan:
            print('In[',i,'] = Nan')
    
    PlugIn = a**2 * In**2;
        #fix the divide by 0 problem
    for i, value in enumerate(PlugIn):
        if value ==0:
            print('PlugIn[',i,'] = ',value)
            PlugIn[i] = 1e-12
    
    I0 = iv(0, PlugIn);
    I1 = iv(1, PlugIn);
    
    Ri = np.log(2 * (1 - np.exp(-0.5 * PlugIn)) / (1 - np.exp(-PlugIn) * I0) );
    
    La = 2 * PlugIn * (np.exp(-PlugIn) * I1) / (1 - np.exp(-PlugIn) * I0) * Ri **(-1);
    
    #fix the divide by 0 problem
    for i, value in enumerate(La):
        if value == 0:
            print('La[',i,'] = ',value)
            La[i] = 1e-8
    
    R = Ri ** (-1 / La)
    return La, R


