import numpy as np
import pandas as pd

# USER LIBRARIES
import multiwell_productivity_index as mpi
from utilities import fit_given_guess, extract_gains

DATA_DIR = "./data/"
OUT_DIR = "./results/"


def init_fit_extract_gains_mc(prod, inj, gains, rng=None, num_threads=1):
    if rng is None:
        rng = np.random.default_rng()
    gains_guess = gains + rng.uniform(-0.1, 0.1, gains.shape)
    gains_guess[gains_guess < 1e-4] = 1e-4
    gains_guess[gains_guess > 1] = 1
    crm_mc = fit_given_guess(prod, inj, gains_guess, num_threads)
    gains_mc = extract_gains(crm_mc, prod.columns, inj.columns)
    return gains_mc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Senlac CRM instances")
    parser.add_argument(
        "output_file", help="Location for output csv file",
    )
    parser.add_argument("-n", help="Number of instances to run", default=10, type=int)
    parser.add_argument(
        "--num_threads", default=1, type=int, help="number of parallel threads to run"
    )
    args = parser.parse_args()

    inj = pd.read_csv(DATA_DIR + "injection_Senlac.csv")
    prod = pd.read_csv(DATA_DIR + "production_Senlac.csv")
    prod.index = pd.date_range("1999-01-31", "2009-12-31", freq="M")

    idx = pd.IndexSlice
    locations = pd.read_csv(
        DATA_DIR + "well_locations_Senlac.csv",
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
    # gains_mc.to_parquet(OUT_DIR + "monte_carlo_gains.parquet")
