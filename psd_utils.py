import numpy as np

NYQUIST = 283.2114

''' model '''

def model(nu, theta , reshape = True):

    nu_max , H, P, tau, alpha, W  = theta

    if reshape:
        nu = np.reshape(nu, (1, -1))
        nu_max = np.reshape(nu_max, (-1, 1))
        H = np.reshape(H, (-1, 1))
        P = np.reshape(P,(-1, 1))
        tau = np.reshape(tau, (-1, 1))
        alpha = np.reshape(alpha, (-1, 1))
    
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha )
    g = excess(nu, nu_max, H)

    model = (W + eta * (b + g))

    return model[0] 

# PSD model per deassis
def damping(nu):
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta

def granulation(nu, P, tau, alpha ):
    granulation =  (P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha))
    if not np.isfinite(granulation).all():
        return nu * np.inf
    return granulation

def excess(nu, nu_max, H):
    FWHM = 0.66 * nu_max ** 0.88
    G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
    return G

def nu_max(M, R, Teff, nu_max_solar = 3090, Teff_solar = 5777):
    return nu_max_solar * M * (R **-2) * ((Teff / Teff_solar)**-0.5)

def delta_nu(M, R, delta_nu_solar = 135.1):
    return delta_nu_solar * (M ** 0.5) * (R ** -1.5)


''' mcmc  '''

def lnlike(logtheta, freq , real_flux):

    H = 10**logtheta[1]
    P = 10**logtheta[2]
    tau = 10**logtheta[3]
    W = 10**logtheta[5]

    theta = [logtheta[0], H , P , tau , logtheta[4] , W ]

    model_flux = model(freq , theta)
    to_sum = np.log(model_flux) + (real_flux / model_flux)
    LnLike = - 1.0 * np.sum( to_sum )

    return LnLike

def lnprior(logtheta):

    nu_max = logtheta[0]
    H = 10**logtheta[1]
    P = 10**logtheta[2]
    tau = 10**logtheta[3]
    alpha = logtheta[4]
    W = 10**logtheta[5]

    if not (3 < nu_max < 150 and 
            150 < H < 2e6 and 
            150 < P < 2e6 and 
            2000 < tau < 100000 and 
            1 < alpha < 6 and 
            0.1 < W < 10000):
        return -np.inf
    else: 
        return 0.0

def lnprob(logtheta, freq , real_flux):
    lp = lnprior(logtheta)
    if not np.isfinite(lp):
        return -np.inf
    return (lp + lnlike(logtheta, freq , real_flux))

