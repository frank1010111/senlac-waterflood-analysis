import numpy as np
import pandas as pd
from pyCRM import CRM
import os

# NUM_CORES = (os.cpu_count() // 2) or 1
idx = pd.IndexSlice


def init_fit_extract_gains_mc(prod, inj, gains, rng=None, num_threads=1):
    if rng is None:
        rng = np.random.default_rng()
    gains_guess = gains + rng.uniform(-0.1, 0.1, gains.shape)
    gains_guess[gains_guess < 1e-4] = 1e-4
    gains_guess[gains_guess > 1] = 1
    crm_mc = fit_given_guess(prod, inj, gains_guess, num_threads)
    gains_mc = extract_gains(crm_mc, prod.columns, inj.columns)
    return gains_mc


def fit_random_instance(production_table: pd.DataFrame, injection_table: pd.DataFrame):
    """
    fit CRM using random inputs

    Parameters
    --------
    production_table: pd.DataFrame
    injection_table: pd.DataFrame

    Outputs
    --------
    crm instance after fitting
    """
    time = production_table.index
    time = (time - time[0]).days.astype(float)
    crm = CRM(primary=True, constraints="up-to one")
    crm.fit(
        production_table.values,
        injection_table.values,
        time.values,
        NUM_CORES,
        random=True,
    )
    return crm


def fit_given_guess(
    production_table: pd.DataFrame,
    injection_table: pd.DataFrame,
    gain_guess: np.ndarray,
    num_threads: int = 1,
) -> CRM:
    """
    fit CRM using a particular gain guess

    Parameters
    --------
    production_table: pd.DataFrame
    injection_table: pd.DataFrame
    gain_guess: 2D np.ndarray, shape: (n_injectors, n_producers)

    Outputs
    --------
    crm instance after fitting
    """
    n_inj = injection_table.shape[1]
    time = production_table.index
    time = (time - time[0]).days.astype(float)
    crm = CRM(primary=True, constraints="up-to one")
    crm.set_rates(production_table.values, injection_table.values, time.values)
    old_guess = crm._get_initial_guess()
    new_guess = [
        np.concatenate([gg, fg[n_inj:]]) for gg, fg in zip(gain_guess, old_guess)
    ]
    crm.fit(
        production_table.values,
        injection_table.values,
        time.values,
        initial_guess=new_guess,
        num_cores=num_threads,
    )
    return crm


def extract_relative_error(crm: CRM, producers: pd.Index):
    residuals = pd.DataFrame(crm.residual(), columns=producers)
    relative_error = (
        residuals.sum() / pd.DataFrame(crm.production, columns=producers).sum()
    )
    return relative_error


def extract_residuals(crm: CRM, producers: pd.Index):
    residuals = pd.DataFrame(crm.residual(), columns=producers).apply(
        lambda x: np.mean(x ** 2), axis="index"
    )
    return residuals


def extract_gains(crm: CRM, producers: pd.Index, injectors: pd.Index):
    gains = pd.DataFrame(crm.gains, producers, injectors)
    taus = pd.DataFrame(crm.tau, producers, injectors)
    well_pairs = (
        pd.concat([gains.stack().rename("Gain"), taus.stack().rename("Tau")], axis=1)
        .reset_index()
        .assign(log_gain=lambda x: np.log10(x.Gain + 1e-6))
    )
    return well_pairs


def extract_primary(crm: CRM, producers: pd.Index):
    primary_fits = pd.DataFrame(
        {"Gain primary": crm.gains_producer, "Tau primary": crm.tau_producer}, producers
    )
    return primary_fits


def injector_producer_unitarrows(locations, injectors, producers):
    out = pd.DataFrame(
        index=pd.MultiIndex.from_product([injectors, producers]),
        columns=["angle", "xn", "yn"],
    )
    for (i, p) in out.index:
        x1, y1 = locations.loc[i, ["X", "Y"]]
        x2, y2 = locations.loc[p, ["X", "Y"]]
        out.loc[idx[i, p], "angle"] = angle(x1, x2, y1, y2)
    out["xn"] = out["angle"].map(np.sin)
    out["yn"] = out["angle"].map(np.cos)
    return out


def arrows(gains_spatial, locations, arrow_factor=1):
    to_join = pd.DataFrame(
        index=gains_spatial.index.drop_duplicates(),
        columns=["angle", "dist", "xn", "yn"],
        dtype="float",
    )
    for (i, p) in to_join.index:
        x1, y1 = locations.loc[i, ["X", "Y"]]
        x2, y2 = locations.loc[p, ["X", "Y"]]
        to_join.loc[idx[i, p], "angle"] = angle(x1, x2, y1, y2)
        to_join.loc[idx[i, p], "dist"] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    to_join["xn"] = to_join["angle"].map(np.sin)
    to_join["yn"] = to_join["angle"].map(np.cos)
    new_gains_df = gains_spatial.join(to_join)
    for axis in ["x", "y"]:
        new_gains_df[axis + "_arrow"] = (
            new_gains_df[axis + "n"] * new_gains_df["Gain"] * arrow_factor
        )
    return new_gains_df.drop(columns=["xn", "yn"])


def angle(x1, x2, y1, y2):
    x_diff = x2 - x1
    y_diff = y2 - y1
    return np.arctan2(x_diff, y_diff)
