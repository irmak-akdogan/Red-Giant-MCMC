#!/bin/env python

import pandas as pd

import argparse
import os

def fit_star(kic_id: int, star_row: pd.Series):
    pass

def fit_stars(data_table: pd.DataFrame):
    for (kic_id, row) in data_table:
        fit_star(kic_id, row)

def parse_args() -> pd.DataFrame:
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

    args = argparser.parse_args()

    if args.data_path is None:
        argparser.error("Need either the --data-path flag or "
                        "$RED_GIANT_DATA_PATH environmental variable to be set")
    
    try:
        full_df = pd.read_csv(args.data_path, index_col="KIC")
    except Exception as e:
        argparser.error(f"Could not read {args.data_path}!\n{e}")

    column_difference = {"KIC", "nu_max", "H", "P", "tau", "alpha"}.difference(full_df.columns)
    if column_difference:
        argparser.error(f"{args.data_path} does not have required columns {column_difference}")

    if args.start_id < 0:
        argparser.error(f"--start-id must be within bounds of data ([0, {len(full_df)}))")

    if args.end_id is None:
        args.end_id = len(full_df)

    if args.end_id > len(full_df) or args.end_id < args.start_id:
        argparser.error(f"--end-id must be within bounds of data ([0, {len(full_df)}])")

    return full_df.iloc[args.start_id:args.end_id]

def main():
    data = parse_args()

    fit_stars(data)

if __name__ == "__main__": main()