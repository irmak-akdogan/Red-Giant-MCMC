#!/bin/env python

import logging

import csv
from sys import stdout
from typing import TextIO
import numpy as np
import pandas as pd

import argparse
import os

from psd_utils import lnprob, grab_data

from scipy.optimize import minimize, OptimizeResult

import emcee

PARAM_NAMES = ['nu_max', 'H', 'P', 'tau', 'alpha', 'W']

def mcmc_debug_plot(sampler: emcee.EnsembleSampler, truths=None):
    import matplotlib.pyplot as plt
    import corner
    chain_fig, axes = plt.subplots(sampler.ndim, 1)
    chain: np.ndarray = sampler.get_chain(discard=100) #type: ignore
    print(chain.shape)

    for dim_idx, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        ax.set_title(name)
        for walker_idx in range(chain.shape[1]):
            ax.plot(chain[:, walker_idx, dim_idx], color='black', alpha=0.1)
    
    chain_fig.savefig("chain-debug-plot.png")
    
    corner_fig = plt.figure()
    corner.corner(sampler.get_chain(flat=True), truths=truths, fig=corner_fig)
    corner_fig.savefig("corner-debug-plot.png")

def do_mcmc(x0_logged_center: np.ndarray, freq, flux):
    ndim = len(x0_logged_center)
    n_walkers = 32

    sampler = emcee.EnsembleSampler(n_walkers, ndim, lnprob, args=(freq, flux))

    SPREAD = 1e-2
    x0_logged = x0_logged_center + SPREAD * np.random.randn(n_walkers, ndim)

    N_STEPS = 200
    sampler.run_mcmc(x0_logged, N_STEPS, progress=True)
    print(sampler.chain.shape)

    mcmc_debug_plot(sampler, truths=x0_logged_center)

    return sampler.get_chain(flat=True)

def process_star(kic_id: str, star_row: pd.Series):
    freq, powers = grab_data(kic_id)

    W0 = np.mean(powers[-40:])

    x0 = np.array(list(star_row[PARAM_NAMES[:-1]]) + [W0])
    
    x0_logged = x0.copy()
    x0_logged[[1, 2, 3, 5]] = np.log10(x0[[1, 2, 3, 5]])
    
    min_res: OptimizeResult = minimize(lambda t: -lnprob(t, freq, powers)/len(freq), x0_logged)
    chain: np.ndarray = do_mcmc(min_res.x, freq, powers) # type: ignore
    cov = np.cov(chain.transpose())

    min_res.x[[1, 2, 3, 5]] = 10**(min_res.x[[1, 2, 3, 5]])

    return min_res, cov

def flatten_covariances(cov: np.ndarray) -> list[int]:
    flat_cov = []
    for i in range(cov.shape[0]):
        for j in range(i, cov.shape[1]):
            flat_cov.append(cov[i, j])
    
    return flat_cov

flat_cov_names = []
for i, l in enumerate(PARAM_NAMES):
    for r in PARAM_NAMES[i:]:
        flat_cov_names.append(f"e_{l}:{r}")

def process_all_stars(data_table: pd.DataFrame, writer):
    writer.writerow(["KIC"] + PARAM_NAMES + ["success", "status"] + flat_cov_names)

    for (kic_id, row) in data_table.iterrows():
        min_res, cov = process_star(kic_id, row) # type: ignore
        
        out_row = [kic_id] + list(min_res.x) + [min_res.success, min_res.status] + flatten_covariances(cov)
        out_row = map(str, out_row)
        
        writer.writerow(out_row)

def parse_args() -> tuple[pd.DataFrame, TextIO]:
    argparser = argparse.ArgumentParser()
    
    env_default = os.environ.get("RED_GIANT_DATA_PATH")
    argparser.add_argument(
        "--data-path",
        type=str,
        default=env_default,
        help="Path to data (defaults to $RED_GIANT_DATA_PATH)"
    )

    argparser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="First row to process."
    )

    argparser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="Last row (exclusive) to process"
    )

    argparser.add_argument(
        "-o", "--out-file",
        type=argparse.FileType("w", bufsize=1),
        default=stdout,
        help="Last row (exclusive) to process"
    )

    args = argparser.parse_args()

    if args.data_path is None:
        argparser.error("Need either the --data-path flag or "
                        "$RED_GIANT_DATA_PATH environmental variable to be set")
    
    try:
        full_df = pd.read_csv(args.data_path, index_col="KIC")
    except Exception as e:
        argparser.error(f"Could not read {args.data_path}!\n{e}")

    column_difference = {"nu_max", "H", "P", "tau", "alpha"}.difference(full_df.columns)
    if column_difference:
        argparser.error(f"{args.data_path} does not have required columns {column_difference}")

    if args.start_id < 0:
        argparser.error(f"--start-id must be within bounds of data ([0, {len(full_df)}))")

    if args.end_id is None:
        args.end_id = len(full_df)

    if args.end_id > len(full_df) or args.end_id < args.start_id:
        argparser.error(f"--end-id must be within bounds of data ([0, {len(full_df)}])")

    return full_df.iloc[args.start_id:args.end_id], args.out_file

def main():
    data, out = parse_args()

    process_all_stars(data, csv.writer(out))

if __name__ == "__main__": main()