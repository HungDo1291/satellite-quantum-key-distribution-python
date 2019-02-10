import numpy as np
import matplotlib.pyplot as plt


def covariance_matrix_satellite_based_entanglement(T_a, T_b, v, chi):
	a = 1 + T_a * (v-1) + chi
	b = 1 + T_b*(v-1) +chi
	c = np.sqrt(T_a *T_b)* np.sqrt(v**2 -1)
	return a, b, c


def f(x):
	y = (x+1)/2* np.log2((x+1)/2) - (x-1)/2 * np.log2((x-1)/2);
	n = len(x)
	for i in range(n):
		if x[i] == 1 :
			y[i] = 0;
			print("warning, taking log of 0 in calculating f(x)")
	return y

def key_rate(a, b, c, method, reference):
	# calculate the sympletic eigenvalues
    z = np.sqrt((a+b)**2 - 4*c**2);
    nu1 = 0.5*(z + b - a);
    nu2 = 0.5*(z - b + a);
    entropy_e = f(nu1) + f(nu2);
    
    if reference == "A":
        if method == "homodyne":
            nu = sqrt(b*(b-c**2/a));
        
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
    filter = key_rate>0;
    key_rate = key_rate*filter;
    return key_rate

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

    a, b, c = covariance_matrix_satellite_based_entanglement(transmissivity_A, transmissivity_B, v, chi)
    key_rate_satellite_based_entanglement = key_rate(a, b, c, method, reference);
    plt.plot(loss_dB, key_rate_satellite_based_entanglement)
    plt.ylabel('Keyrate (bits per pulse)')
    plt.xlabel('Loss (dB)')
    plt.title('Key rate of satellite-based entanglement with fixed attenuation channel')
    plt.yscale('log')
    plt.xlim(0,20)
    plt.show()
	
if __name__ == "__main__":
    key_rate_plots_fixed_attenuation()