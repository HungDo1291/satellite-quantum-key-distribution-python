import numpy as np
import matplotlib.pyplot as plt
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
    transmission_coefficient = np.logspace(-2.0, 0.0, num=1000)
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
    k_dv_pdc = key_rate_DV.key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                    Y_0A, Y_0B, e_0,e_d, q, f, 'PDC_II', lambd)
    
    #hybrid
    r= r
    k_dv_hybrid = key_rate_DV.key_rate_DV(transmissivity_A, transmissivity_B, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, 'hybrid', r)
    
    plt.figure()
    plt.plot(loss_dB, key_rate_cv, label = "CV Direct Transmission scheme")
    plt.plot(loss_dB, k_dv_hybrid, label = "DV Hybrid, Source on Alice")   
    plt.plot(loss_dB, k_dv_pdc, label = "DV from PDC II, Source on Alice") 
    
    plt.ylabel('Keyrate (bits per pulse)')
    plt.xlabel('Loss (dB)')
    plt.title('Key rate with source on Alice side and fixed-attenuation channel')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig('figures/fixed_attenuation_source_on_Alice.eps', format='eps', dpi=1000)
    plt.savefig('figures/fixed_attenuation_source_on_Alice.jpg', format='jpg', dpi=1000)

    plt.show()
    
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
    
    