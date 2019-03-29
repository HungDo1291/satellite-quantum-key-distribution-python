import numpy as np
#import matplotlib
#matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
import matplotlib.pyplot as plt
#for surface plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import turbulence_transmission_coefficient as turbulence
import key_rate_CV
import key_rate_DV
import hybrid_entanglement_swapping as hybrid


##########################
# CV PLOTS
#####################################

#######################################################
# PLOT KEY RATE FOR FIXED ATTENUATION #################
##############################################    

def key_rate_plots_fixed_attenuation():
    """
    """
    transmission_coefficient = np.logspace(-1.0, 0.0, num=1000)
    transmissivity_A = transmission_coefficient ** 2
    transmissivity_B = transmissivity_A
    loss_dB = -10 * np.log10(transmissivity_A * transmissivity_B)

    chi = 0.02
    method = 'homodyne'
    reference = 'B'
    r = 1
    v = np.cosh(r)

    a, b, c = key_rate_CV.covariance_matrix_satellite_based_entanglement(transmissivity_A, transmissivity_B, v, chi)
    key_rate_satellite_based_entanglement = key_rate_CV.key_rate_CV(a, b, c, method, reference);
    plt.plot(loss_dB, key_rate_satellite_based_entanglement)
    plt.ylabel('Keyrate (bits per pulse)')
    plt.xlabel('Loss (dB)')
    plt.title('Key rate of satellite-based entanglement with fixed attenuation channel')
    plt.yscale('log')
    plt.xlim(0,20)
    plt.show()
    
def key_rate_plots_fixed_attenuation_source_on_Alice():
    """
    """
    transmission_coefficient = np.logspace(-10.0, 0.0, num=10000)
    transmissivity_B = transmission_coefficient ** 2
    transmissivity_A = np.ones(len(transmissivity_B))
    loss_dB = -10 * np.log10(transmissivity_A * transmissivity_B)
    
    #CV parameters
    chi = 0.02
    method = 'homodyne'
    reference = 'B'
    r = 2.395
    v = np.cosh(2*r)

    a, b, c = key_rate_CV.covariance_matrix(transmissivity_A, transmissivity_B, v, chi, 'direct transmission')
    key_rate_cv = key_rate_CV.key_rate_CV(a, b, c, method, reference);
    
    a, b, c = key_rate_CV.covariance_matrix(transmissivity_A, transmissivity_B, v, chi, 'entanglement swapping')
    key_rate_cv_mdi = key_rate_CV.key_rate_CV(a, b, c, method, reference);
    
    #dv parameters
    f = 1.22
    q = 1/2
    e_0 = 1/2
    e_d = 0.015

    Y_0A =6.02e-6
    Y_0B = Y_0A
    detection_efficiency = 0.145
    
    #PDC_II
    mu = np.linspace(0.01,0.5,int(0.5/0.01));
    lambd = mu/2
    Q_dv_pdc, E_dv_pdc, k_dv_pdc = key_rate_DV.key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                    Y_0A, Y_0B, e_0,e_d, q, f, 'PDC_II', lambd)
    
    #hybrid
    loss_on = 'loss on DV entanglement'
    Q_hybrid_dv, E_hybrid_dv, k_hybrid_dv, P_n_dv = key_rate_DV.key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, 'hybrid', loss_on)
    
    loss_on = 'loss on CV entanglement'
    Q_hybrid_cv, E_hybrid_cv, k_hybrid_cv, P_n_cv = key_rate_DV.key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, 'hybrid', loss_on)
    loss_dB = -10 * np.log10(transmissivity_A * transmissivity_B)
    
    #plot keyrate
    fig, ax=plt.subplots()
    plt.plot(loss_dB, key_rate_cv,'.', label = "CV Ent from Direct Transmission")    
    plt.plot(loss_dB, key_rate_cv_mdi,'.', label = "CV Ent from CV-CV swapping")
    plt.plot(loss_dB, k_hybrid_dv,'.', label = "DV Ent from hybrid DV-CV swapping, loss on DV")  
    plt.plot(loss_dB, k_hybrid_cv,'.', label = "DV Ent from hybrid DV-CV swapping, loss on CV")  
    plt.plot(loss_dB, k_dv_pdc, '.',label = "DV Ent from PDC II") 
    plt.grid()
    plt.yscale('log')


    fig.tight_layout()
    zoomed = False
    if zoomed == True:
        plt.xlim([0,3])
        plt.xticks(np.arange(0,4,1)) # set this to zoom in the small loss region
        plt.ylim([1e-4,1e1])
        plt.yticks([1e-4,1e-3,1e-2,1e-1,1e0,1e1])
        plt.savefig('figures/fixed_attenuation_source_on_Alice_zoomed.eps', format='eps', dpi=1000)
        plt.savefig('figures/fixed_attenuation_source_on_Alice_zoomed.jpg', format='jpg', dpi=1000)
    else:
        plt.ylabel('Keyrate (bits per pulse)')
        plt.xlabel('Optical link loss (dB)')
        plt.title('Key rate from fixed-attenuation channel')
        lgd= plt.legend(loc='upper right', bbox_to_anchor=(0.9, -0.14),ncol=1)
        plt.savefig('figures/fixed_attenuation_source_on_Alice.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig('figures/fixed_attenuation_source_on_Alice.jpg', format='jpg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()
    
    #plot QBER
    print('E_hybrid_dv : ',E_hybrid_dv)
    print('Q_hybrid_dv : ',Q_hybrid_dv)
    fig, ax=plt.subplots()
    plt.plot(loss_dB, E_hybrid_cv,'r', label = "DV Ent from hybrid DV-CV swapping, loss in CV")
    plt.plot(loss_dB, E_hybrid_dv,'r--', label = "DV Ent from hybrid DV-CV swapping, loss in DV") 
    plt.plot(loss_dB, E_dv_pdc, label = "DV Ent from PDC II") 


    #plt.yscale('log')

    fig.tight_layout()
    plt.grid()
    zoomed = False
    if zoomed == True:
        plt.xlim([0,1])
        plt.savefig('figures/QBER_fixed_attenuation_source_on_Alice_zoomed.eps', format='eps', dpi=1000)
        plt.savefig('figures/QBER_fixed_attenuation_source_on_Alice_zoomed.jpg', format='jpg', dpi=1000)
    else:    
    
        plt.ylabel('QBER')
        plt.xlabel('Optical link loss (dB)')
        lgd= plt.legend(loc='upper right', bbox_to_anchor=(1.1, -0.2),ncol=1)
        plt.savefig('figures/QBER_fixed_attenuation_source_on_Alice.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.savefig('figures/QBER_fixed_attenuation_source_on_Alice.jpg', format='jpg', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.show()    
    
    #find median loss where loss_dB = 15dB
    min_index = np.argmin(np.abs(loss_dB - 100))
    median_index = np.argmin(np.abs(loss_dB - 3))
    print(median_index, min_index)
    
    length_loss = len(loss_dB)
    print('length_loss = ', length_loss)
    
    #loss on cv
    fig, ax1 = plt.subplots()
    ax1.plot(P_n_cv[:, int(length_loss-1)], 'k', label ='T = 1 ')
#    for i in range(0,length_loss,int(length_loss/20)):
#        #ax1.plot(P_n_dv[:, i], 'r:')
#        ax1.plot(P_n_cv[:, i], 'k:')
    #ax1.plot(P_n_dv[:, median_index], 'ro', label = 'loss on DV entanglement, 3dB')
    #ax1.plot(P_n_dv[:, min_index], 'rx', label = 'loss on DV entanglement, 40dB')

    ax1.plot(P_n_cv[:, median_index], 'b', label = 'T = 0.5 (3dB)')
    ax1.plot(P_n_cv[:, min_index],'r', label = 'T=1e-10 (100dB)')
    
    plt.xlim([-1,10])
    plt.ylabel('Probability of n-photon pairs P(n)')
    plt.xlabel('Photon-number n')
    plt.grid()
    plt.legend()
    plt.savefig('figures/P_N_lossCV.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    fig.show()

    #loss on dv
    print('P(n) loss on dv = ', P_n_dv)
    fig, ax1 = plt.subplots()
    ax1.plot(P_n_dv[:, int(length_loss-1)], 'k', label ='T = 1 ')
#    for i in range(0,length_loss,int(length_loss/20)):
#        #ax1.plot(P_n_dv[:, i], 'r:')
#        ax1.plot(P_n_cv[:, i], 'k:')
    #ax1.plot(P_n_dv[:, median_index], 'ro', label = 'loss on DV entanglement, 3dB')
    #ax1.plot(P_n_dv[:, min_index], 'rx', label = 'loss on DV entanglement, 40dB')

    ax1.plot(P_n_dv[:, median_index], 'b', label = 'T = 0.5 (3dB)')
    ax1.plot(P_n_dv[:, min_index],'r', label = 'T=1e-10 (100dB)')

    #ax1.plot(P_n_cv[:, int(length_loss-1)],'k.')
    plt.xlim([-1,10])
    plt.ylim([10e-31,1])
    plt.yscale('log')
    plt.ylabel('Probability of n-photon pairs P(n)')
    plt.xlabel('Photon-number n')

    plt.grid()
    plt.legend()
    plt.savefig('figures/P_N_lossDV.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    fig.show()
    
    lambd_average = 0.1
    n = np.array(np.arange(51))
    P_n_pdc = (n+1)*lambd_average**n / (1+lambd_average)**(n+2)
    
    fig, ax1 = plt.subplots()
    ax1.plot(P_n_pdc, 'k', label ='T = 1 ')
    plt.xlim([-1,10])
    plt.ylabel('Probability of n-photon pairs P(n)')
    plt.xlabel('Photon-number n')
    plt.grid()
    plt.legend()
    fig.show()

"""
    fig, ax2 = plt.subplots()    
    ax2.plot(P_n_dv[:, int(length_loss-1)], 'ko', label = 'no loss')
    for i in range(0,length_loss,int(length_loss/10)):
        ax1.plot(P_n_dv[:, i], 'r:')
        ax1.plot(P_n_cv[:, i], 'b:')
    ax2.plot(P_n_dv[:, median_index], 'ro', label = 'loss on DV entanglement, 3dB')
    ax2.plot(P_n_dv[:, min_index], 'rx', label = 'loss on DV entanglement, 40dB')
    ax2.plot(P_n_cv[:, median_index], 'bo', label = 'loss on CV entanglement, 3dB')
    ax2.plot(P_n_cv[:, min_index],'bx', label = 'loss on CV entanglement, 40dB')

    ax2.plot(P_n_cv[:, int(length_loss-1)],'k.')

    plt.yscale('log')  
    
    plt.grid()
    plt.legend()
    fig.show()
    
""" 

#    print(P_n_dv.shape, P_n_cv.shape, P_n_dv-P_n_cv)
#        # PLOTTING
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    a = np.array([[1e-1,1e-2,1e-3]])
#    b = a
#    c = a.T.dot(b)
#    X, Y = np.meshgrid(np.array(range(51)), loss_dB)
#    surf = ax.plot_surface(Y,X,P_n_dv-P_n_cv, cmap=cm.coolwarm,\
#-                       linewidth=0, antialiased=False)
#    surf2 = ax.plot_surface(a,b,c, cmap=cm.coolwarm,\
#                       linewidth=0, antialiased=False)
#    ax.set_xlabel('g')
#    ax.set_ylabel('r')
#    ax.set_zlabel('E_{LN}')
#    ax.set_zscale('log')
#    # Add a color bar which maps values to colors.
#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.show()
    
    
#################################################### 
# PLOT KEY RATE FOR SATELLITE FADING CHANNEL
####################################################    
def key_rate_plots_satellite_fading_channel(generate_transmissivity= False):
    sigma_a = np.linspace(0.01 ,12, 20)
    n_sigma_a = len(sigma_a)

    L= 500000 #distance from satellite to Earth : 500 km
    wavelength = 780*1e-9
    W0 = 0.12
    aperture_radius = 1
    num_points = int(1e4); #typecast to integer inorder to avoid errors
    turbulence_model = 'circular'
    
    if generate_transmissivity == True: # generate transmissivity
        transmissivity_1, transmissivity_2 = turbulence.generate_transmissivity(sigma_a, turbulence_model,\
                                                     W0 , L, wavelength, aperture_radius,  num_points )
    else: #load transmissivity from file
        """ !!! still need to save and load sigma_a as well !!!"""
        transmissivity_1  = np.fromfile('transmissivity_1_'+ turbulence_model+'.dat', dtype = float).reshape([n_sigma_a, num_points])
        transmissivity_2  = np.fromfile('transmissivity_1_'+ turbulence_model+'.dat', dtype = float).reshape([n_sigma_a, num_points])

    #cv parameters
    v=60
    chi = 0.02 #excess noise at the receiver
    measurement_method = 'homodyne'
    reconciliation_ref = 'B' #reversed reconciliation
    
    t_cv1, k_cv1, mean_loss_cv1, mean_k_cv1 = \
        key_rate_CV.mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a, \
        num_points, 'satellite based entanglement',v, chi,measurement_method, reconciliation_ref)

    t_cv2, k_cv2, mean_loss_cv2, mean_k_cv2 = \
        key_rate_CV.mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a, \
        num_points, 'entanglement swapping',v, chi,measurement_method, reconciliation_ref)

    t_cv3, k_cv3, mean_loss_cv3, mean_k_cv3 = \
        key_rate_CV.mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a, \
        num_points, 'direct transmission',v, chi,measurement_method, reconciliation_ref)
        
    #dv parameters
    f = 1.22
    q = 1/2
    e_0 = 1/2
    e_d = 0.015

    Y_0A =6.02e-6
    Y_0B = Y_0A
    detection_efficiency = 0.145
    
    #hybrid source
    r=1
    t_hybrid, k_hybrid, mean_loss_hybrid, mean_k_hybrid = \
    key_rate_DV.mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,n_sigma_a,   num_points, \
                                  'hybrid', r, detection_efficiency,Y_0A, Y_0B, e_0,e_d, q, f)
    #source = 'PDC_II'
    mu = np.linspace(0.01,0.5,int(0.5/0.01));
    lambd = mu/2;
    t_pdc, k_pdc, mean_loss_pdc, mean_k_pdc = key_rate_DV.mean_key_rate_fading_channel(transmissivity_1, transmissivity_2,\
                                                                                       n_sigma_a,  num_points, \
                              'PDC_II', lambd , detection_efficiency,Y_0A, Y_0B, e_0,e_d, q, f)

    fig, ax = plt.subplots()
    plt.plot(mean_loss_cv3, mean_k_cv3, 'o',label = 'CV direct transmission')
    plt.plot(mean_loss_cv1, mean_k_cv1, 'o', label = 'CV satellite based entanglement')
    plt.plot(mean_loss_cv2, mean_k_cv2, 'o', label = 'CV entanglement swapping')
    plt.plot(mean_loss_hybrid, mean_k_hybrid, label = 'DV hybrid_circular_r=1')
    plt.plot(mean_loss_pdc, mean_k_pdc, label = 'DV pdc_circular_mu = 0.01:0.01:0.5')

    plt.legend(loc ='best')
    #plt.title('CV '+turbulence_model +' '+ measurement_method + ' '+reconciliation_ref+' v=60')
    plt.title(turbulence_model+' approximation')
    plt.xlabel("mean loss (dB)")
    plt.ylabel("mean key rate")
    plt.yscale('log')
    plt.grid()
    plt.savefig(turbulence_model+'.eps', format='eps', dpi=1000)
    plt.savefig(turbulence_model+'.jpg', format='jpg', dpi=1000)
    fig.show()
    
    
if __name__ == "__main__":
    #test functions
    # turbulence.test_turbulence()
    #key_rate_DV.test_DV_key_rate()
    #hybrid.test_hybrid()
    
    # key_rate_plots_fixed_attenuation()
    #key_rate_plots_satellite_fading_channel()
    key_rate_plots_fixed_attenuation_source_on_Alice()
    
    