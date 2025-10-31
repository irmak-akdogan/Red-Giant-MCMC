from numba import njit
import numpy as np

NYQUIST = 283.2114


''' model '''

@njit
def damping(nu):
    """Compute damping (apodization) factor."""
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta

@njit
def granulation(nu, P, tau, alpha):
    """Compute granulation component of PSD."""
    denom = 1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha
    result = P / denom
    return result

@njit
def excess(nu, nu_max, H):
    """Compute Gaussian oscillation excess."""
    FWHM = 0.66 * nu_max ** 0.88
    sigma_sq = FWHM ** 2 / (4 * np.log(2))
    G = H * np.exp(-(nu - nu_max)**2 / sigma_sq)
    return G

@njit
def model_single(nu, nu_max, H, P, tau, alpha, W):
    """
    Compute PSD model for a single parameter set (1D frequency array).
    
    Parameters:
    -----------
    nu : 1D array
        Frequency array
    nu_max, H, P, tau, alpha, W : float
        Model parameters (scalars)
    
    Returns:
    --------
    1D array of model values
    """
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha)
    g = excess(nu, nu_max, H)
    
    result = W + eta * (b + g)
    return result

@njit
def model_batch(nu, nu_max_arr, H_arr, P_arr, tau_arr, alpha_arr, W_arr):
    """
    Compute PSD model for multiple parameter sets (batched).
    
    Parameters:
    -----------
    nu : 1D array (n_freq,)
        Frequency array
    nu_max_arr, H_arr, P_arr, tau_arr, alpha_arr, W_arr : 1D arrays (n_models,)
        Arrays of model parameters
    
    Returns:
    --------
    2D array (n_models, n_freq) of model values
    """
    n_models = len(nu_max_arr)
    n_freq = len(nu)
    result = np.empty((n_models, n_freq))
    
    for i in range(n_models):
        result[i, :] = model_single(nu, nu_max_arr[i], H_arr[i], P_arr[i], 
                                     tau_arr[i], alpha_arr[i], W_arr[i])
    
    return result

def model(nu, theta, reshape=True):
    """
    Wrapper function for backward compatibility.
    
    Parameters:
    -----------
    nu : array-like
        Frequency values
    theta : array-like
        [nu_max, H, P, tau, alpha, W]
    reshape : bool
        If True, treats inputs as potentially batched (legacy behavior)
        If False, treats as single parameter set
    
    Returns:
    --------
    array of model values
    """
    nu_max, H, P, tau, alpha, W = theta
    
    # Convert to numpy arrays
    nu = np.asarray(nu)
    
    if reshape:
        # Handle batched inputs (for MCMC posterior draws)
        nu_max = np.atleast_1d(nu_max)
        H = np.atleast_1d(H)
        P = np.atleast_1d(P)
        tau = np.atleast_1d(tau)
        alpha = np.atleast_1d(alpha)
        W = np.atleast_1d(W)
        
        if len(nu_max) == 1:
            # Single parameter set
            return model_single(nu, nu_max[0], H[0], P[0], tau[0], alpha[0], W[0])
        else:
            # Multiple parameter sets
            return model_batch(nu, nu_max, H, P, tau, alpha, W)
    else:
        # Single parameter set, no reshape
        return model_single(nu, float(nu_max), float(H), float(P), 
                           float(tau), float(alpha), float(W))

# def model(nu, theta , reshape = True):

#     nu_max , H, P, tau, alpha, W  = theta

#     if reshape:
#         nu = np.reshape(nu, (1, -1))
#         nu_max = np.reshape(nu_max, (-1, 1))
#         H = np.reshape(H, (-1, 1))
#         P = np.reshape(P,(-1, 1))
#         tau = np.reshape(tau, (-1, 1))
#         alpha = np.reshape(alpha, (-1, 1))
    
#     eta = damping(nu)
#     b = granulation(nu, P, tau, alpha )
#     g = excess(nu, nu_max, H)

#     model = (W + eta * (b + g))

#     return model[0] 

# # PSD model per deassis
# def damping(nu):
#     eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
#     return eta

# def granulation(nu, P, tau, alpha ):
#     granulation =  (P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha))
#     if not np.isfinite(granulation).all():
#         return nu * np.inf
#     return granulation

# def excess(nu, nu_max, H):
#     FWHM = 0.66 * nu_max ** 0.88
#     G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
#     return G

# def nu_max(M, R, Teff, nu_max_solar = 3090, Teff_solar = 5777):
#     return nu_max_solar * M * (R **-2) * ((Teff / Teff_solar)**-0.5)

# def delta_nu(M, R, delta_nu_solar = 135.1):
#     return delta_nu_solar * (M ** 0.5) * (R ** -1.5)


