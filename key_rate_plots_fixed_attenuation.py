import numpy as np
import matplotlib.pyplot as plt
import turbulence_transmission_coefficient as turbulence
import key_rate_CV
import key_rate_DV


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
    
    
#################################################### 
# PLOT KEY RATE FOR SATELLITE FADING CHANNEL
####################################################    
def key_rate_plots_satellite_fading_channel():
    sigma_a = np.logspace(-3 ,1, 20)
    n_sigma_a = len(sigma_a)

    L= 500000 #distance from satellite to Earth : 500 km
    wavelength = 780*1e-9
    W0 = 0.12
    aperture_radius = 1
    num_points = int(1e4); #typecast to integer inorder to avoid errors
    turbulence_model = 'circular'
    
    #cv parameters
    v=60
    chi = 0.02 #excess noise at the receiver
    measurement_method = 'homodyne'
    reconciliation_ref = 'B' #reversed reconciliation
    transmissivity_1  = np.fromfile('transmissivity_1_circular.dat', dtype = float).reshape([n_sigma_a, num_points])
    transmissivity_2  = np.fromfile('transmissivity_1_circular.dat', dtype = float).reshape([n_sigma_a, num_points])
    print(transmissivity_1.shape)
    
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
    plt.plot(mean_loss_cv3, mean_k_cv3, label = 'CV direct transmission')
    plt.plot(mean_loss_cv1, mean_k_cv1, label = 'CV satellite based entanglement')
    plt.plot(mean_loss_cv2, mean_k_cv2, label = 'CV entanglement swapping')
    plt.plot(mean_loss_hybrid, mean_k_hybrid, label = 'DV hybrid_circular_r=1')
    plt.plot(mean_loss_pdc, mean_k_pdc, label = 'DV pdc_circular_mu = 0.01:0.01:0.5')

    plt.legend(loc =9, bbox_to_anchor=(0.5, -0.3))
    #plt.title('CV '+turbulence_model +' '+ measurement_method + ' '+reconciliation_ref+' v=60')
    plt.title('circular beam wandering approximation')
    plt.xlabel("mean loss (dB)")
    plt.ylabel("mean key rate")
    plt.yscale('log')
    plt.grid()
    plt.savefig('circular.eps', format='eps', dpi=1000)
    plt.savefig('circular.jpg', format='jpg', dpi=1000)
    fig.show()
    
def DV_key_rate_plots_satellite_fading_channel():
    
    sigma_a = np.logspace(-3 ,1, 20)
    n_sigma_a = len(sigma_a)

    L= 500000 #distance from satellite to Earth : 500 km
    wavelength = 780*1e-9
    W0 = 0.12
    aperture_radius = 1
    num_points = int(1e4); #typecast to integer inorder to avoid errors
    turbulence_model = 'circular'
    
    # generate transmissivity
#    transmissivity_1, transmissivity_2 = turbulence.generate_transmissivity(sigma_a, turbulence_model,\
#                                                     W0 , L, wavelength, aperture_radius,  num_points )
    transmissivity_1  = np.fromfile('transmissivity_1_circular.dat', dtype = float).reshape([n_sigma_a, num_points])
    transmissivity_2  = np.fromfile('transmissivity_1_circular.dat', dtype = float).reshape([n_sigma_a, num_points])
    print(transmissivity_1.shape)
    
    
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
    
    fig1, ax1 = plt.subplots()
    plt.plot(mean_loss_hybrid, mean_k_hybrid, label = 'hybrid_circular_r=1')
    plt.plot(mean_loss_pdc, mean_k_pdc, label = 'pdc_circular_mu = 0.01:0.01:0.5')

    #plt.plot(CV_mean_loss_dB_60_el, CV_mean_key_rate_60_el, label = 'elliptic '+ measurement_method + ' '+reconciliation_ref+' v=60')
    #plt.plot(CV_mean_loss_dB_60_hete_el, CV_mean_key_rate_60_hete_el, label = 'elliptic heterodyne ' +reconciliation_ref +' v=60')
    plt.legend()
    plt.xlabel("mean loss (dB)")
    plt.ylabel("mean key rate")
    plt.yscale('log')
    plt.title('Key rate from DV entanglement')
    plt.grid()
    plt.savefig('DV_circular.eps', format='eps', dpi=1000)
    plt.savefig('DV_circular.jpg', format='jpg', dpi=1000)
    fig1.show()
    

    
    

#################################################
#   TEST FUNCTIONS
#################################################
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
      
    _, tc1 = turbulence.circular_beam_wandering_trans_coef(sigma_a[4], W0 , L, \
        wavelength, aperture_radius,  num_simulation_points, num_simulation_points, test = True)
    _, tc2 = turbulence.elliptic_model_trans_coef(sigma_a[4], W0 , L, \
        wavelength, aperture_radius,  num_simulation_points, test = True)
    
    fig, ax = plt.subplots()
    for i in range(n_sigma):
        _, tc2 = turbulence.elliptic_model_trans_coef(sigma_a[i], W0 , L, wavelength, aperture_radius,  num_simulation_points)    
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
    K_hybrid = key_rate_DV.key_rate_DV(transmissivity1, transmissivity2, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
    
    source = 'PDC_II'
    mu = np.linspace(0.01,0.5,int(0.5/0.01));
    lambd = mu/2;
    source_parameter = lambd
    K_PDCII = key_rate_DV.key_rate_DV(transmissivity1, transmissivity2, detection_efficiency,\
                Y_0A, Y_0B, e_0,e_d, q, f, source, source_parameter)
    
    fig1, ax1 = plt.subplots()
    plt.plot(loss_dB, K_hybrid, label = 'hybrid')
    print(loss_dB.shape, K_PDCII.shape, transmissivity1.shape)
    plt.plot(loss_dB, K_PDCII, label = 'PDC type II')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == "__main__":
    #key_rate_plots_fixed_attenuation()
    #test_turbulence()
    #test_DV_key_rate()

    key_rate_plots_satellite_fading_channel()
    #DV_key_rate_plots_satellite_fading_channel()
    
    