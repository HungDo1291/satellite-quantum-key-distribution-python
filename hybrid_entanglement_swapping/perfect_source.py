# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:46:00 2019

@author: twink
"""
#libraries
import numpy as np
import matplotlib.pyplot as plt

#for surface plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time


#for latex in matplotlib
import matplotlib
#for font in matplotlib
matplotlib.rcParams['text.usetex'] = True
font = {'family' : 'serif',
  'weight' : 'bold',
  'size' : 16}
matplotlib.rc('font', **font)

#for other libraries in this project
import functions as hybrid_functions

#constants
SQUEEZING = 2.395;
k_MAX = 100;
EFFICIENCY_AT_DV_SOURCE = 1;

##########################################
# TEST
##########################################

def test_hybrid():
    print("-------------")
    print("test_hybrid()")
    print("-------------")
    r = SQUEEZING
    k_max = k_MAX    
    T = np.logspace(-1.0, 0.0, num=50); 
    g = np.linspace(0.1, 2, 100);

    #type cast array of array so that we can transpose and perform outer product
    T = np.array([T])
    g = np.array([g]).T
    print('shape of gT is: ',g.dot(T).shape)
    print('shape of g+T is: ',(g+T).shape)
    
    start_time = time.time()
    
    tau_total, gamma, E_LN = calculate_ELN(r,g,T, k_max)
    
    print("--- Run time of calculate_ELN is %s seconds ---" % (time.time() - start_time))
    
    fig, ax=plt.subplots()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(T, g)
    surf = ax.plot_surface(X,Y,E_LN, cmap=cm.coolwarm,\
                       linewidth=0, antialiased=False)

    ax.set_xlabel('T')
    ax.set_ylabel('g')
    ax.set_zlabel('$E_{LN}$')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    figure_name = '../figures/perfect_source/E_LN_all'
    plt.savefig(figure_name + '.eps', format='eps', dpi=1000,  bbox_inches='tight')#  bbox_extra_artists=(lgd,),
    plt.savefig(figure_name + '.jpg', format='jpg', dpi=1000,  bbox_inches='tight')
    

    #fix g =1
    g = np.array([1])
    tau_total, gamma, E_LN = calculate_ELN(r,g,T, k_max)
    
    fig, ax=plt.subplots()
    plt.plot(T[0,:],E_LN[0,:])

    ax.set_xlabel('T')
    ax.set_ylabel('$E_{LN}$')

    plt.show()
    
    figure_name = '../figures/perfect_source/E_LN_g1'
    plt.savefig(figure_name + '.eps', format='eps', dpi=1000,  bbox_inches='tight')#  bbox_extra_artists=(lgd,),
    plt.savefig(figure_name + '.jpg', format='jpg', dpi=1000,  bbox_inches='tight')

    
    ###########
    P_0 = hybrid_functions.probability_n_photon_pair(0, k_max,  gamma, g)
    print(P_0.shape)
    P_1 = hybrid_functions.probability_n_photon_pair(1, k_max, gamma, g) 
    P_2 = hybrid_functions.probability_n_photon_pair(2, k_max,  gamma, g)
    P_3 = hybrid_functions.probability_n_photon_pair(3, k_max,  gamma, g)
    P_4 = hybrid_functions.probability_n_photon_pair(4, k_max,  gamma, g)
    P_50 = hybrid_functions.probability_n_photon_pair(50, k_max,  gamma, g)
    print('P(0,1,2,3,4,50) = ', P_0[0,49], P_1[0,49], P_2[0,49], P_3[0,49], P_4[0,49], P_50[0,49])
    
    #######################
    n_max = 20
    n_array = np.array([range(n_max)]).T
    print(n_array)
    P_n = np.zeros([n_max, 50])
    print(P_n[4,:].shape)
    for n in range(n_max):
        P_n[n,:] = hybrid_functions.probability_n_photon_pair(n, k_max,  gamma, g)[0]
#        
    fig, ax=plt.subplots()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(T,n_array)
    surf = ax.plot_surface(X,Y,P_n, cmap=cm.coolwarm,\
                       linewidth=0, antialiased=False)

    ax.set_xlabel('T')
    ax.set_ylabel('n')
    ax.set_zlabel('$P_n$')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    figure_name = '../figures/perfect_source/P_n_g1'
    plt.savefig(figure_name + '.eps', format='eps', dpi=1000,  bbox_inches='tight')#  bbox_extra_artists=(lgd,),
    plt.savefig(figure_name + '.jpg', format='jpg', dpi=1000,  bbox_inches='tight')
    
def test_surf():
    print("------------------------------")
    print(" Test surf function in python ")
    print("------------------------------")
    a = np.array([[1,2,3,4,5]])
    b = np.array([[2,4]]).T
    c = b.dot(a)
    print(a,b,c)
    fig, ax=plt.subplots()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(a, b)
    surf = ax.plot_surface(X,Y, c, cmap=cm.coolwarm,\
                       linewidth=0, antialiased=False)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

##########################################################
    # FUNCTION
##########################################################
def teleportation_attenuation(r,g, T):
    print("teleportation_attenuation")
    temp1 = (1 + g**2)*(np.exp(2*r) + np.exp(-2*r))
    temp2 = 2*g*(np.exp(2*r) - np.exp(-2*r))
    tau_teleportation = (temp1 - temp2)/(4*g**2)
    g_teleportation = g

    tau_attenuation = (1 - T)/ (2*T)
    g_attenuation = np.sqrt(T)
    
    tau_total = tau_teleportation + tau_attenuation/g_teleportation**2
    g_total = g_teleportation * g_attenuation
    return tau_total, g_total
    
def attenuation_teleportation(r,g, T):
    print("attenuation_teleportation")
    temp1 = (1 + g**2)*(np.exp(2*r) + np.exp(-2*r))
    temp2 = 2*g*(np.exp(2*r) - np.exp(-2*r))
    tau_teleportation = (temp1 - temp2)/(4*g**2)
    g_teleportation = g

    tau_attenuation = (1 - T)/ (2*T)
    g_attenuation = np.sqrt(T)
    
    tau_total =  tau_attenuation  + tau_teleportation/(g_attenuation**2)
    g_total = g_teleportation * g_attenuation
    return tau_total, g_total

def calculate_gamma(g, tau):
    gamma = g**2*(2*tau + 1) ;
    return gamma
    
def sum_negative_eigen_values(k_max, gamma, g):

    lenghth_k = k_max+2;
    summation = 0#np.zeros(gamma.shape);

    for i in range(lenghth_k):
        k = i-1; #when i runs from 0 to k_max+1, while k runs from -1 to k_max
        a = hybrid_functions.f_a( gamma, k, g);
        b = hybrid_functions.f_b( gamma, k, g);
        c = hybrid_functions.f_c( gamma, k);
        lambd = 0.5 * ( a+c - np.sqrt((a - c)**2 + 4 * b**2) );
        summation +=  ( np.abs(lambd) - lambd ) ;
    return summation



def calculate_ELN( r,g,T, k_max):
    tau_total, g_total = attenuation_teleportation(r,g, T)
    gamma = calculate_gamma(g_total, tau_total)

    summation = sum_negative_eigen_values(k_max, gamma, g_total);
    #print ('final sum of -ev = ', summation)
    E_LN = np.log2(1+summation);
    return tau_total, gamma, E_LN


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
    plt.close("all")
    test_hybrid()
    #test_surf()