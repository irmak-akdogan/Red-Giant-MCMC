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
niter = 5_000
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

    # use get_chain() instead of flatchain for compatibility
    flat_chain = sampler.get_chain(flat=True, discard=discard)
    cov = np.corrcoef(flat_chain.T)

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

def corner_plot(sampler, truths=None):
    fig = corner.corner(sampler.get_chain(flat=True, discard = discard), labels=["numax", "log10(H)", "log10(P)", "log10(tau)", "alpha", "log10(W)"], show_titles=True, title_fmt=".3f", bins=30, truths=truths)

    # Add text box with initial values if truths are provided
    if truths is not None:
        param_names = ["numax", "log10(H)", "log10(P)", "log10(tau)", "alpha", "log10(W)"]
        
        # Get the chain to calculate means and stds
        flat_chain = sampler.get_chain(flat=True, discard=discard)
        means = np.mean(flat_chain, axis=0)
        stds = np.std(flat_chain, axis=0)
        
        # Calculate z-scores (how many standard deviations from mean)
        z_scores = (truths - means) / stds
        
        # Build text with initial values and z-scores
        truth_text = "Initial Values vs Posterior:\n"
        for name, val, z in zip(param_names, truths, z_scores):
            truth_text += f"{name} = {val:.3f} ({z:+.2f}σ)\n"
        
        # Add text box in the upper right corner of the figure
        fig.text(0.98, 0.98, truth_text, 
                transform=fig.transFigure, 
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig

def parameter_estimates(sampler):
    estimates = []
    # use get_chain() instead of flat_samples for compatibility
    flat_chain = sampler.get_chain(flat=True, discard=discard)
    for i in range(flat_chain.shape[1]):
        mcmc = np.percentile(flat_chain[:, i], [16, 50, 84])
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
    freq, powers = data
    
    # Calculate W as the mean of the last 20 µHz of the PSD
    mask = freq >= (NYQUIST - 20)
    W_initial = np.mean(powers[mask])
    
    theta = np.append(row[['nu_max', 'H', 'P', 'tau', 'alpha']], W_initial)
    logtheta = np.array([theta[0], np.log10(theta[1]), np.log10(theta[2]), np.log10(theta[3]), theta[4], np.log10(theta[5])])
    initial = np.array(logtheta)

    bump = initial * 0.0001
    p0 = [initial + bump * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler, pos, prob, state = main(p0, data, position=progress_id, progress_queue=progress_queue)
    cov = get_cov(sampler, show=False)

    return index , sampler, cov, logtheta

def run(start, end, r = 0):
    rows = list(table.iloc[start:end].iterrows())
    progress_queue = mp.Manager().Queue()

    # prepare output directory before launching workers so workers can report
    os.makedirs("runs", exist_ok=True)
    outdir = f"runs/run {r}"
    os.makedirs(outdir, exist_ok=True)

    # summary CSV to append per-star results as they complete
    summary_path = os.path.join(outdir, "summary.csv")
    header_written = os.path.exists(summary_path)

    # will collect results incrementally as stars finish
    results = []
    idx_to_future = {}

    with ProcessPoolExecutor(max_workers= cpu_count()) as executor:
        # submit all jobs and keep a mapping from index -> future
        for index, row in rows:
            future = executor.submit(run_one_star, index, row, index, progress_queue)
            idx_to_future[index] = future

        # create 2 tqdm bars per star, labeled with the actual index
        burn_bars = {idx: tqdm(total=1000, desc=f"Burn-in {idx}", position=2*i)
                    for i, idx in enumerate([idx for idx, _ in rows])}
        prod_bars = {idx: tqdm(total=niter, desc=f"Production {idx}", position=2*i+1)
                    for i, idx in enumerate([idx for idx, _ in rows])}

        finished = 0
        # listen to progress queue and when a star is done, fetch and save its result
        while finished < len(rows):
            msg_type, idx, value = progress_queue.get()
            if msg_type == "burn":
                if idx in burn_bars:
                    burn_bars[idx].n = value
                    burn_bars[idx].refresh()
            elif msg_type == "prod":
                if idx in prod_bars:
                    prod_bars[idx].n = value
                    prod_bars[idx].refresh()
            elif msg_type == "done":
                finished += 1

                # retrieve the corresponding future and its result (should be ready)
                future = idx_to_future.get(idx)
                if future is None:
                    print(f"Warning: no future found for finished star {idx}")
                    continue

                try:
                    idx_res, sampler, cov, logtheta = future.result()

                    # compute parameter estimates and append a summary row
                    try:
                        ests = parameter_estimates(sampler)
                        # ests: list of (median, minus, plus) for each param
                        cols = [
                            'index', 'kic', 'sampler_file', 'cov_file',
                            'nu_max_med', 'nu_max_minus', 'nu_max_plus',
                            'H_med', 'H_minus', 'H_plus',
                            'P_med', 'P_minus', 'P_plus',
                            'tau_med', 'tau_minus', 'tau_plus',
                            'alpha_med', 'alpha_minus', 'alpha_plus',
                            'W_med', 'W_minus', 'W_plus'
                        ]

                        medians = []
                        for m, lo, hi in ests:
                            medians.extend([m, lo, hi])

                        kic_val = int(table.iloc[idx_res]["KIC"]) if idx_res < len(table) else ''
                        # we no longer save samplers or chain files
                        row = {
                            'index': idx_res,
                            'kic': kic_val,
                            'sampler_file': '',
                            'cov_file': f"{idx_res:05d}_cov.npy",
                        }

                        # add medians and errors
                        param_names = ['nu_max', 'H', 'P', 'tau', 'alpha', 'W']
                        for i, name in enumerate(param_names):
                            row[f"{name}_med"] = medians[3*i]
                            row[f"{name}_minus"] = medians[3*i+1]
                            row[f"{name}_plus"] = medians[3*i+2]

                        # save only cov matrix (no sampler or chain)
                        np.save(f"{outdir}/{idx_res:05d}_cov.npy", cov)

                        # append to CSV (using pandas to handle header automatically)
                        df_row = pd.DataFrame([row])
                        df_row.to_csv(summary_path, mode='a', header=not header_written, index=False)
                        header_written = True

                        # append lightweight result (index, medians list, cov)
                        results.append((idx_res, medians, cov))

                        # generate plots immediately before discarding sampler
                        try:
                            # covariance matrix plot
                            cov_result, fig = get_cov(sampler, show=True)
                            fig.savefig(f"{outdir}/{idx_res:05d}_cov_matrix.png")
                            plt.close(fig)

                            # corner plot
                            fig = corner_plot(sampler, truths=logtheta)
                            fig.savefig(f"{outdir}/{idx_res:05d}_corner.png")
                            plt.close(fig)

                            # convergence plot
                            fig = check_convergence(sampler)
                            fig.savefig(f"{outdir}/{idx_res:05d}_convergence.png")
                            plt.close(fig)

                            # posteriors plot
                            fig = posteriors(sampler, idx_res)
                            fig.savefig(f"{outdir}/{idx_res:05d}_posteriors.png")
                            plt.close(fig)

                            print(f"Plots saved for star {idx_res}")
                        except Exception as e:
                            print(f"Warning: couldn't generate plots for star {idx_res}: {e}")

                        # explicitly drop sampler to avoid keeping large objects in memory
                        try:
                            del sampler
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Warning: couldn't process results for star {idx_res}: {e}")

                    # mark progress bars as complete for this star
                    if idx in burn_bars:
                        burn_bars[idx].n = burn_bars[idx].total
                        burn_bars[idx].refresh()
                    if idx in prod_bars:
                        prod_bars[idx].n = prod_bars[idx].total
                        prod_bars[idx].refresh()

                except Exception as e:
                    print(f"Error retrieving result for star {idx}: {e}")

    # sort and save aggregated arrays (final snapshot)
    if len(results) == 0:
        print("No results to save.")
        return

    results.sort(key=lambda x: x[0])
    # results now contains tuples (index, medians_list, cov)
    indices = [r[0] for r in results]
    medians_list = [r[1] for r in results]
    cov_list = [r[2] for r in results]

    # save lightweight aggregated outputs
    np.save(f"{outdir}/indices.npy", np.array(indices))
    np.save(f"{outdir}/medians.npy", np.array(medians_list))
    np.save(f"{outdir}/cov_matrices.npy", np.array(cov_list))

def grab_plots(r, offset = 0, samplers = None): 

    if samplers is None: 
        samplers = np.load(f"runs/run {r}/samplers.npy", allow_pickle=True)

    for i in range(len(samplers)):

        s = samplers[i]
        cov, fig = get_cov(s, show = True)
        fig.savefig(f"runs/run {r}/{i+ offset:05d}_cov_matrix.png")
        plt.close(fig)

        fig = corner_plot(s)
        fig.savefig(f"runs/run {r}/{i+ offset:05d}_corner.png")
        plt.close(fig)

        fig = check_convergence(s)
        fig.savefig(f"runs/run {r}/{i+ offset:05d}_convergence.png")
        plt.close(fig)

        fig = posteriors(s, i+ offset)
        fig.savefig(f"runs/run {r}/{i+ offset:05d}_posteriors.png")
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
    # grab_plots() is disabled since we no longer save samplers
    # grab_plots(r = args.run_id, offset = args.start )
