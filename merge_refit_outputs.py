#!/bin/env python

import pandas as pd

import os
from pathlib import Path

PATH = Path("refit_output")

dfs = []
for file in PATH.rglob("refit_*.csv"):
    dfs.append(pd.read_csv(file, index_col="KIC"))

concatenated_dfs = pd.concat(dfs)


ref_df = pd.read_csv(os.environ.get("RED_GIANT_DATA_PATH", ""), index_col="KIC")

reordered_df = concatenated_dfs.loc[ref_df.index]

reordered_df.to_csv(PATH / "merged_refit.csv")