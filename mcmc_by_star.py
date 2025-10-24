import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from tqdm import tqdm

import emcee
import corner


# Constants
NYQUIST = 283.2114
NWALKERS = 15
NITER = 5_000
NDIM = 6
DISCARD = NITER // 5
BURN_IN_ITERATIONS = 1000

# Solar reference values
NU_MAX_SOLAR = 3090
TEFF_SOLAR = 5777
DELTA_NU_SOLAR = 135.1

# Data loading
data_path = os.environ.get("RED_GIANT_DATA_PATH", "/Users/student/Desktop/RedGiant_MCMC/merged_data.csv")
table = pd.read_csv(data_path)


# ============================================================================
# Model Functions
# ============================================================================

def model(nu, theta, reshape=True):
    """
    Compute the power spectral density model.
    
    Args:
        nu: Frequency array
        theta: Model parameters [nu_max, H, P, tau, alpha, W]
        reshape: Whether to reshape arrays for broadcasting
        
    Returns:
        Model PSD values
    """
    nu_max, H, P, tau, alpha, W = theta

    if reshape:
        nu = np.reshape(nu, (1, -1))
        nu_max = np.reshape(nu_max, (-1, 1))
        H = np.reshape(H, (-1, 1))
        P = np.reshape(P, (-1, 1))
        tau = np.reshape(tau, (-1, 1))
        alpha = np.reshape(alpha, (-1, 1))
    
    eta = damping(nu)
    b = granulation(nu, P, tau, alpha)
    g = excess(nu, nu_max, H)

    model_psd = W + eta * (b + g)

    return model_psd[0] 

def damping(nu):
    """Calculate the damping factor (sinc function)."""
    eta = np.sinc((1 / 2) * (nu / NYQUIST)) ** 2
    return eta


def granulation(nu, P, tau, alpha):
    """Calculate the granulation component of the PSD."""
    granulation_val = P / (1 + (2 * np.pi * tau * 1e-6 * nu) ** alpha)
    if not np.isfinite(granulation_val).all():
        return nu * np.inf
    return granulation_val


def excess(nu, nu_max, H):
    """Calculate the Gaussian excess component around nu_max."""
    FWHM = 0.66 * nu_max ** 0.88
    G = H * np.exp(-(nu - nu_max)**2 / (FWHM ** 2 / (4 * np.log(2))))
    return G


def nu_max(M, R, Teff):
    """Calculate expected nu_max from stellar parameters."""
    return NU_MAX_SOLAR * M * (R ** -2) * ((Teff / TEFF_SOLAR) ** -0.5)


def delta_nu(M, R):
    """Calculate expected delta_nu from stellar parameters."""
    return DELTA_NU_SOLAR * (M ** 0.5) * (R ** -1.5)


# ============================================================================
# MCMC Functions
# ============================================================================

def lnlike(logtheta, freq, real_flux):
    """Calculate the log-likelihood."""
    H = 10**logtheta[1]
    P = 10**logtheta[2]
    tau = 10**logtheta[3]
    W = 10**logtheta[5]

    theta = [logtheta[0], H, P, tau, logtheta[4], W]

    model_flux = model(freq, theta)
    to_sum = np.log(model_flux) + (real_flux / model_flux)
    LnLike = -1.0 * np.sum(to_sum)

    return LnLike


def lnprior(logtheta):
    """Calculate the log-prior."""
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


def lnprob(logtheta, freq, real_flux):
    """Calculate the log-probability (prior + likelihood)."""
    lp = lnprior(logtheta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(logtheta, freq, real_flux)

def main(p0, data, position, progress_queue):
    """
    Run MCMC sampling with burn-in and production phases.
    
    Args:
        p0: Initial positions for walkers
        data: Tuple of (frequency, power) data
        position: Progress bar position identifier
        progress_queue: Queue for reporting progress
        
    Returns:
        Tuple of (sampler, final_positions, probabilities, state)
    """
    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, lnprob, args=data)

    # Burn-in phase
    pos, prob, state = None, None, None
    for i, (pos, prob, state) in enumerate(sampler.sample(p0, iterations=BURN_IN_ITERATIONS)):
        if i % 10 == 0:  # Update every 10 steps
            progress_queue.put(("burn", position, i + 1))
    sampler.reset()

    # Production phase
    for i, (pos, prob, state) in enumerate(sampler.sample(pos, iterations=NITER)):
        if i % 100 == 0:
            progress_queue.put(("prod", position, i + 1))

    progress_queue.put(("done", position, None))

    return sampler, pos, prob, state


# ============================================================================
# Utility Functions
# ============================================================================

def grab_data(kic_num):
    """
    Download and process light curve data for a given KIC number.
    
    Args:
        kic_num: Kepler Input Catalog number
        
    Returns:
        Tuple of (frequency, power) arrays
    """
    search = lk.search_lightcurve('KIC ' + str(kic_num), author='Kepler')
    lc = search.download_all().stitch(lambda x: x.normalize('ppm'))
    valid = np.isfinite(lc.flux_err) & (lc.flux_err > 0)

    pd = lc[valid].to_periodogram(
        normalization='psd', 
        minimum_frequency=1, 
        maximum_frequency=NYQUIST
    )

    freq = pd.frequency.to_value()
    powers = pd.power.to_value()

    return (freq, powers)


# ============================================================================
# Plotting Functions
# ============================================================================

def corner_plot(sampler, truths=None):
    """
    Create a corner plot of the MCMC posterior distributions.
    
    Args:
        sampler: emcee sampler object
        truths: Optional initial parameter values to overlay
        
    Returns:
        matplotlib figure
    """
    fig = corner.corner(
        sampler.get_chain(flat=True, discard=DISCARD), 
        labels=["numax", "log10(H)", "log10(P)", "log10(tau)", "alpha", "log10(W)"], 
        show_titles=True, 
        title_fmt=".3f", 
        bins=30, 
        truths=truths
    )

    # Add text box with initial values if truths are provided
    if truths is not None:
        param_names = ["numax", "log10(H)", "log10(P)", "log10(tau)", "alpha", "log10(W)"]
        
        # Get the chain to calculate means and stds
        flat_chain = sampler.get_chain(flat=True, discard=DISCARD)
        means = np.mean(flat_chain, axis=0)
        stds = np.std(flat_chain, axis=0)
        
        # Calculate z-scores (how many standard deviations from mean)
        z_scores = (truths - means) / stds
        
        # Build text with initial values and z-scores
        truth_text = "Initial Values vs Posterior:\n"
        for name, val, z in zip(param_names, truths, z_scores):
            truth_text += f"{name} = {val:.3f} ({z:+.2f}σ)\n"
        
        # Add text box in the upper right corner of the figure
        fig.text(
            0.98, 0.98, truth_text, 
            transform=fig.transFigure, 
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    return fig


def parameter_estimates(sampler):
    """
    Calculate median and error estimates for each parameter.
    
    Args:
        sampler: emcee sampler object
        
    Returns:
        List of (median, lower_error, upper_error) tuples
    """
    estimates = []
    flat_chain = sampler.get_chain(flat=True, discard=DISCARD)
    for i in range(flat_chain.shape[1]):
        mcmc = np.percentile(flat_chain[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        estimates.append((mcmc[1], q[0], q[1]))
    return estimates


def check_convergence(sampler):
    """
    Plot parameter chains to check MCMC convergence.
    
    Args:
        sampler: emcee sampler object
        
    Returns:
        matplotlib figure
    """
    chain = sampler.get_chain(flat=False, discard=DISCARD)
    param_names = ['numax', 'H', 'P', 'tau', 'alpha', 'W']

    fig = plt.figure() 
    
    for i, name in enumerate(param_names):
        fig.add_subplot(6, 1, i + 1) 
        plt.plot(chain[:, :, i], color='purple', alpha=0.3)
        plt.ylabel(name)
    
    plt.xlabel('Iteration')
    plt.suptitle("How Parameters Change over Iterations")

    return fig 

def get_cov(sampler, show=False):
    """
    Calculate and optionally plot the correlation matrix of parameters.
    
    Args:
        sampler: emcee sampler object
        show: If True, create a plot; if False, just return the matrix
        
    Returns:
        Correlation matrix (and figure if show=True)
    """
    samples = sampler.get_chain(flat=True, discard=DISCARD)
    cov = np.corrcoef(samples.T)

    if not show:
        return cov

    discrete_viridis = ListedColormap(plt.cm.seismic(np.linspace(0, 1, 7)))
    param_names = ["nu_max", "H", "P", "tau", "alpha", "W"]

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cov, cmap=discrete_viridis, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names, rotation=90)
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_yticklabels(param_names)

    ax.set_title("Correlation Coefficient Matrix", pad=20)

    return cov, fig


def posteriors(sampler, j, n=20):
    """
    Plot posterior draws against observed data.
    
    Args:
        sampler: emcee sampler object
        j: Index in the table for this star
        n: Number of posterior draws to plot
        
    Returns:
        matplotlib figure
    """
    name = int(table.iloc[j]["KIC"])
    chain = sampler.get_chain(flat=True, discard=DISCARD)
    H = 10**(chain[:, 1])
    P = 10**(chain[:, 2])
    tau = 10**(chain[:, 3])
    W = 10**(chain[:, 5])
    samples = np.column_stack([chain[:, 0], H, P, tau, chain[:, 4], W])

    # The data
    data = grab_data(name)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(data[0], data[1], color="black", alpha=0.7)

    x = np.logspace(np.log10(1), np.log10(NYQUIST), 100)

    # Posterior draws
    for i in np.random.choice(samples.shape[0], size=n, replace=False):
        draw = samples[i]
        model_draw = model(x, draw)
        plt.plot(x, model_draw, color="orange", alpha=0.1)

    # Initial guess
    theta = np.append(table.iloc[j][['nu_max', 'H', 'P', 'tau', 'alpha']], 10)
    plt.plot(x, model(x, theta), color="red", alpha=1)

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


# ============================================================================
# Multiprocessing Functions
# ============================================================================

def run_one_star(index, row, progress_id, progress_queue):
    """
    Run MCMC for a single star.
    
    Args:
        index: Star index in the table
        row: Row from the data table for this star
        progress_id: ID for progress reporting
        progress_queue: Queue for reporting progress
        
    Returns:
        Tuple of (index, sampler, covariance_matrix, initial_logtheta)
    """
    data = grab_data(int(row["KIC"]))
    freq, powers = data
    
    # Calculate W as the mean of the last 20 µHz of the PSD
    mask = freq >= (NYQUIST - 20)
    W_initial = np.mean(powers[mask])
    
    theta = np.append(row[['nu_max', 'H', 'P', 'tau', 'alpha']], W_initial)
    logtheta = np.array([
        theta[0], 
        np.log10(theta[1]), 
        np.log10(theta[2]), 
        np.log10(theta[3]), 
        theta[4], 
        np.log10(theta[5])
    ])
    initial = np.array(logtheta)

    bump = initial * 0.0001
    p0 = [initial + bump * np.random.randn(NDIM) for _ in range(NWALKERS)]

    sampler, pos, prob, state = main(p0, data, position=progress_id, progress_queue=progress_queue)
    cov = get_cov(sampler, show=False)

    return index, sampler, cov, logtheta

def run(start, end, r=0):
    """
    Run MCMC for a range of stars with multiprocessing.
    
    Args:
        start: Starting index in the table
        end: Ending index (exclusive)
        r: Run identifier for output directory
    """
    rows = list(table.iloc[start:end].iterrows())
    progress_queue = mp.Manager().Queue()

    # Prepare output directory before launching workers
    os.makedirs("runs", exist_ok=True)
    outdir = f"runs/run {r}"
    os.makedirs(outdir, exist_ok=True)

    # Summary CSV to append per-star results as they complete
    summary_path = os.path.join(outdir, "summary.csv")
    header_written = os.path.exists(summary_path)

    # Will collect results incrementally as stars finish
    results = []
    idx_to_future = {}

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        # Submit all jobs and keep a mapping from index -> future
        for index, row in rows:
            future = executor.submit(run_one_star, index, row, index, progress_queue)
            idx_to_future[index] = future

        # Create 2 tqdm bars per star, labeled with the actual index
        burn_bars = {
            idx: tqdm(total=BURN_IN_ITERATIONS, desc=f"Burn-in {idx}", position=2 * i)
            for i, idx in enumerate([idx for idx, _ in rows])
        }
        prod_bars = {
            idx: tqdm(total=NITER, desc=f"Production {idx}", position=2 * i + 1)
            for i, idx in enumerate([idx for idx, _ in rows])
        }

        finished = 0
        # Listen to progress queue and when a star is done, fetch and save its result
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

                # Retrieve the corresponding future and its result (should be ready)
                future = idx_to_future.get(idx)
                if future is None:
                    print(f"Warning: no future found for finished star {idx}")
                    continue

                try:
                    idx_res, sampler, cov, logtheta = future.result()

                    # Compute parameter estimates and append a summary row
                    try:
                        ests = parameter_estimates(sampler)
                        
                        kic_val = int(table.iloc[idx_res]["KIC"]) if idx_res < len(table) else ''
                        
                        # Build row for CSV
                        row = {
                            'index': idx_res,
                            'kic': kic_val,
                            'sampler_file': '',
                            'cov_file': f"{idx_res:05d}_cov.npy",
                        }

                        # Add medians and errors
                        param_names = ['nu_max', 'H', 'P', 'tau', 'alpha', 'W']
                        for i, name in enumerate(param_names):
                            row[f"{name}_med"] = ests[i][0]
                            row[f"{name}_minus"] = ests[i][1]
                            row[f"{name}_plus"] = ests[i][2]

                        # Save only cov matrix (no sampler or chain)
                        np.save(f"{outdir}/{idx_res:05d}_cov.npy", cov)

                        # Append to CSV
                        df_row = pd.DataFrame([row])
                        df_row.to_csv(summary_path, mode='a', header=not header_written, index=False)
                        header_written = True

                        # Append lightweight result (index, medians list, cov)
                        medians = [ests[i][j] for i in range(6) for j in range(3)]
                        results.append((idx_res, medians, cov))

                        # Generate plots immediately before discarding sampler
                        try:
                            # Covariance matrix plot
                            cov_result, fig = get_cov(sampler, show=True)
                            fig.savefig(f"{outdir}/{idx_res:05d}_cov_matrix.png")
                            plt.close(fig)

                            # Corner plot
                            fig = corner_plot(sampler, truths=logtheta)
                            fig.savefig(f"{outdir}/{idx_res:05d}_corner.png")
                            plt.close(fig)

                            # Convergence plot
                            fig = check_convergence(sampler)
                            fig.savefig(f"{outdir}/{idx_res:05d}_convergence.png")
                            plt.close(fig)

                            # Posteriors plot
                            fig = posteriors(sampler, idx_res)
                            fig.savefig(f"{outdir}/{idx_res:05d}_posteriors.png")
                            plt.close(fig)

                            print(f"Plots saved for star {idx_res}")
                        except Exception as e:
                            print(f"Warning: couldn't generate plots for star {idx_res}: {e}")

                        # Explicitly drop sampler to avoid keeping large objects in memory
                        del sampler
                        
                    except Exception as e:
                        print(f"Warning: couldn't process results for star {idx_res}: {e}")

                    # Mark progress bars as complete for this star
                    if idx in burn_bars:
                        burn_bars[idx].n = burn_bars[idx].total
                        burn_bars[idx].refresh()
                    if idx in prod_bars:
                        prod_bars[idx].n = prod_bars[idx].total
                        prod_bars[idx].refresh()

                except Exception as e:
                    print(f"Error retrieving result for star {idx}: {e}")

    # Sort and save aggregated arrays (final snapshot)
    if len(results) == 0:
        print("No results to save.")
        return

    results.sort(key=lambda x: x[0])
    indices = [r[0] for r in results]
    medians_list = [r[1] for r in results]
    cov_list = [r[2] for r in results]

    # Save lightweight aggregated outputs
    np.save(f"{outdir}/indices.npy", np.array(indices))
    np.save(f"{outdir}/medians.npy", np.array(medians_list))
    np.save(f"{outdir}/cov_matrices.npy", np.array(cov_list))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCMC by star index range.")
    parser.add_argument("start", type=int, help="Start index")
    parser.add_argument("end", type=int, help="End index (exclusive)")
    parser.add_argument("run_id", type=int, help="Run number")

    args = parser.parse_args()
    print(f"Running stars {args.start}:{args.end} for run {args.run_id}")
    run(args.start, args.end, r=args.run_id)
