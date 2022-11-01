"""Script for running CRM fits on Senlac data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pywaterflood import multiwellproductivity as mpi

from senlac import init_fit_extract_gains_mc

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data/"
OUT_DIR = ROOT_DIR / "results/"


def main():
    """Run Senlac CRM matches from MPI first guesses."""
    parser = argparse.ArgumentParser(description="Run Senlac CRM instances")
    parser.add_argument(
        "output_file",
        help="Location for output csv file",
    )
    parser.add_argument("-n", help="Number of instances to run", default=10, type=int)
    parser.add_argument(
        "--num_threads", default=1, type=int, help="number of parallel threads to run"
    )
    args = parser.parse_args()

    inj = pd.read_csv(DATA_DIR / "injection_Senlac.csv")
    prod = pd.read_csv(DATA_DIR / "production_Senlac.csv")
    prod.index = pd.date_range("1999-01-31", "2009-12-31", freq="M")
    locations = pd.read_csv(
        DATA_DIR / "well_locations_Senlac.csv",
        skiprows=2,
        names=["Well", "X", "Y"],
        index_col=0,
    )
    locations["Type"] = locations.index.map(
        lambda x: "Producer" if x[0] == "P" else "Injector"
    )
    locations_translated = mpi.translate_locations(locations, "X", "Y", "Type")
    locations_translated[["X", "Y"]] += 100
    x_e, y_e = locations_translated[["X", "Y"]].max() + 100

    gains_mpi = -mpi.calc_gains_homogeneous(locations_translated, x_e, y_e)
    rng = np.random.default_rng()
    gains_mc = pd.concat(
        [
            init_fit_extract_gains_mc(
                prod, inj, gains_mpi.values, rng, num_threads=args.num_threads
            )
            for i in range(args.n)
        ]
    )
    gains_mc.to_csv(args.output_file)


if __name__ == "__main__":
    main()
