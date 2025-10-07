import lightkurve as lk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import emcee
import corner

import os
import argparse

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count

data_path = os.environ.get("RED_GIANT_DATA_PATH", "/Users/student/Desktop/RedGiant_MCMC/merged_data.csv")
table = pd.read_csv(data_path)

NYQUIST = 283.2114
nwalkers = 15
niter = 5000
ndim = 6
discard = niter // 5 

# to do 
# use https://www.pymc.io/projects/examples/en/latest/introductory/api_quickstart.html ? 
# 


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

def main(p0, data, position, progress_queue):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    # --- Burn-in ---
    pos, prob, state = None, None, None
    for i, (pos, prob, state) in enumerate(sampler.sample(p0, iterations=1000)):
        if i % 10 == 0:  # update every 10 steps
            progress_queue.put(("burn", position, i+1))
    sampler.reset()

    # --- Production ---
    for i, (pos, prob, state) in enumerate(sampler.sample(pos, iterations=niter)):
        if i % 100 == 0:
            progress_queue.put(("prod", position, i+1))

    progress_queue.put(("done", position, None))

    return sampler, pos, prob, state


''' utils'''

def get_cov(sampler, show = False):

    cov = np.corrcoef(sampler.flatchain.T)

    if not show: 
        return cov 

    discrete_viridis = ListedColormap(plt.cm.seismic(np.linspace(0, 1, 7)))
    param_names = ["nu_max", "H", "P", "tau", "alpha", "W"]

    plt.matshow(cov, cmap=discrete_viridis)
    plt.clim(-1, 1)
    plt.colorbar()

    plt.xticks(ticks=np.arange(len(param_names)), labels=param_names, rotation=90)
    plt.yticks(ticks=np.arange(len(param_names)), labels=param_names)

    plt.title("Correlation Coefficient Matrix")

    return cov 

def grab_data(kic_num):
    search = lk.search_lightcurve('KIC ' + str(kic_num), author='Kepler')
    lc = search.download_all().stitch(lambda x: x.normalize('ppm'))
    NYQUIST = 283.2114
    valid = np.isfinite(lc.flux_err) & (lc.flux_err > 0)

    pd = lc[valid].to_periodogram(normalization = 'psd', minimum_frequency = 1, maximum_frequency = NYQUIST)

    freq = pd.frequency.to_value()
    powers = pd.power.to_value()

    return (freq, powers)

''' plots from sampler '''

def corner_plot(sampler):
    fig = corner.corner(sampler.get_chain(flat=True, discard = discard), labels=["numax", "H", "P", "tau", "alpha", "W"], show_titles=True, title_fmt=".3f", bins=30)

    return fig

def parameter_estimates(sampler):
    estimates = []
    for i in range(sampler.flat_samples.shape[1]):
        mcmc = np.percentile(sampler.flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        estimates.append((mcmc[1], q[0], q[1]))
    return estimates

def check_convergence(sampler):

    chain = sampler.get_chain(flat=False, discard = discard)

    fig = plt.figure() 
    fig.add_subplot(6, 1, 1) 

    plt.plot(chain[:, :, 0], color='purple', alpha=0.3)
    plt.ylabel('numax')

    fig.add_subplot(6, 1, 2) 
    plt.plot(chain[:, :, 1], color='purple', alpha=0.3)
    plt.ylabel('H')

    fig.add_subplot(6, 1, 3) 
    plt.plot(chain[:, :, 2], color='purple', alpha=0.3)
    plt.ylabel('P')

    fig.add_subplot(6, 1, 4) 
    plt.plot(chain[:, :, 3], color='purple', alpha=0.3)
    plt.ylabel('tau')

    fig.add_subplot(6, 1, 5) 
    plt.plot(chain[:, :, 4], color='purple', alpha=0.3)
    plt.ylabel('alpha')

    fig.add_subplot(6, 1, 6) 
    plt.plot(chain[:, :, 5], color='purple', alpha=0.3)
    plt.ylabel('W')

    plt.xlabel('Iteration')
    plt.suptitle("How Parameters Change over Iterations")

    return fig 

def get_cov(sampler, show=False):

    samples = sampler.get_chain(flat=True, discard=discard)
    cov = np.corrcoef(samples.T)

    if not show:
        return cov

    discrete_viridis = ListedColormap(plt.cm.seismic(np.linspace(0, 1, 7)))
    param_names = ["nu_max", "H", "P", "tau", "alpha", "W"]

    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.matshow(cov, cmap=discrete_viridis, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=90)
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_yticklabels(param_names)

    ax.set_title("Correlation Coefficient Matrix", pad=20)

    return cov, fig

def posteriors(sampler , j, n = 20): 

    name = int(table.iloc[j]["KIC"])
    chain  = sampler.get_chain(flat = True, discard = discard)
    H = 10**(chain[:,1])
    P = 10**(chain[:,2])
    tau = 10**(chain[:,3])
    W = 10**(chain[:,5])
    samples = np.column_stack([chain[:,0], H, P, tau, chain[:,4], W])

    #the data
    data = grab_data(name)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data[0], data[1], color="black", alpha=0.7)

    x = np.logspace(np.log10(1), np.log10(NYQUIST), 100)

    #posterior draws
    for i in np.random.choice(samples.shape[0], size=n, replace=False):
        draw = samples[i]
        model_draw = model(x, draw)
        plt.plot(x, model_draw, color="orange", alpha=0.1)

    #initial guess
    theta = np.append(table.iloc[j][['nu_max', 'H', 'P', 'tau', 'alpha']], 10)
    plt.plot(x, model(x, theta), color = "red", alpha=1)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Frequency (µHz)")
    plt.ylabel("Power")

    plt.legend(
        handles=[
            Line2D([0], [0], color="orange", alpha=1, label="Posterior Draws"),
            Line2D([0], [0], color="red", alpha=1, label="Initial Guess"),
            Line2D([0], [0], color="black", alpha=1, label="Observed Data"),
        ]
    )
    plt.title(f"Posterior Draws vs Observed Data for {name}")

    return fig 

''' multiprocessing '''

def run_one_star(index, row, progress_id, progress_queue):

    data = grab_data(int(row["KIC"]))
    theta = np.append(row[['nu_max', 'H', 'P', 'tau', 'alpha']], 10)
    logtheta = np.array([theta[0], np.log10(theta[1]), np.log10(theta[2]), np.log10(theta[3]), theta[4], np.log10(theta[5])])
    initial = np.array(logtheta)

    bump = initial * 0.01
    p0 = [initial + bump * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler, pos, prob, state = main(p0, data, position=progress_id, progress_queue=progress_queue)
    cov = get_cov(sampler, show=False)

    return index , sampler, cov

def run(start, end, r = 0):

    rows = list(table.iloc[start:end].iterrows())
    progress_queue = mp.Manager().Queue()

    with ProcessPoolExecutor(max_workers= cpu_count()) as executor:
        futures = {
            executor.submit(run_one_star, index, row, index, progress_queue): index
            for index, row in rows
        }

        # create 2 tqdm bars per star, labeled with the actual index
        burn_bars = {idx: tqdm(total=1000, desc=f"Burn-in {idx}", position=2*i)
                    for i, idx in enumerate([idx for idx, _ in rows])}
        prod_bars = {idx: tqdm(total=niter, desc=f"Production {idx}", position=2*i+1)
                    for i, idx in enumerate([idx for idx, _ in rows])}

        finished = 0
        while finished < len(rows):
            msg_type, idx, value = progress_queue.get()
            if msg_type == "burn":
                burn_bars[idx].n = value
                burn_bars[idx].refresh()
            elif msg_type == "prod":
                prod_bars[idx].n = value
                prod_bars[idx].refresh()
            elif msg_type == "done":
                finished += 1

        results = []
        for f in futures:
            idx, sampler, cov = f.result()
            results.append((idx, sampler, cov))

    results.sort(key=lambda x: x[0])
    indices, sampler_list, cov_list = zip(*results)
    
    os.makedirs("runs", exist_ok=True)
    outdir = f"runs/run {r}"
    os.makedirs(outdir, exist_ok=True)

    np.save(f"{outdir}/samplers.npy", np.array(sampler_list))
    np.save(f"{outdir}/cov_matrices.npy", np.array(cov_list))
    np.save(f"{outdir}/indices.npy", np.array(indices))
    np.save(f"{outdir}/results.npy", np.array(results, dtype=object))

def grab_plots(r, offset = 0, samplers = None): 

    if samplers is None: 
        samplers = np.load(f"runs/run {r}/samplers.npy", allow_pickle=True)

    for i in range(len(samplers)):

        s = samplers[i]
        cov, fig = get_cov(s, show = True)
        fig.savefig(f"runs/run {r}/{i+ offset}_cov_matrix.png")
        plt.close(fig)

        fig = corner_plot(s)
        fig.savefig(f"runs/run {r}/{i+ offset}_corner.png")
        plt.close(fig)

        fig = check_convergence(s)
        fig.savefig(f"runs/run {r}/{i+ offset}_convergence.png")
        plt.close(fig)

        fig = posteriors(s, i+ offset)
        fig.savefig(f"runs/run {r}/{i+ offset}_posteriors.png")
        plt.close(fig)

        print(f"Star {i+ offset} is done.")

    return samplers

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run MCMC by star index range.")
    parser.add_argument("start", type=int, help="Start index")
    parser.add_argument("end", type=int, help="End index (exclusive)")
    parser.add_argument("run_id", type=int, help="Run number (default: 0)")

    args = parser.parse_args()
    print(f"Running stars {args.start}:{args.end} for run {args.run_id}")
    run(args.start, args.end,r = args.run_id )
    grab_plots(r = args.run_id, offset = args.start )
