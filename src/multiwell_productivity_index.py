import numpy as np
import pandas as pd
import scipy.linalg as sl

idx = pd.IndexSlice


def calc_gains_homogeneous(
    locations: pd.DataFrame, x_e: float, y_e: float
) -> pd.DataFrame:
    """Provides gains using multiwell productivity index methodology,
    following Valkó and others, 2000, "Development and application of the multiwell
    productivity index (MPI)," SPEJ.

    Inputs
    --------
    locations: X and Y coordinates of well locations and Type in {Producer, Injector}
    x_e: extent of flow area in x direction
    y_e: extent of flow area in y direction

    Output
    --------
    Lambda: a matrix with connectivities, index: producers, columns: injectors
    """
    locations = locations.copy()
    locations[["X", "Y"]] /= x_e
    y_D = y_e / x_e
    A_prod = calc_influence_matrix(locations, y_D, "prod").astype(float)
    A_conn = calc_influence_matrix(locations, y_D, "conn").astype(float)
    A_prod_inv = sl.inv(A_prod.values)
    term1 = A_prod_inv / np.sum(A_prod_inv)
    term2 = np.ones_like(A_prod_inv) @ A_prod_inv @ A_conn.values - 1
    term3 = A_prod_inv @ A_conn.values
    Lambda = term1 @ term2 - term3
    return pd.DataFrame(Lambda, index=A_prod.index, columns=A_conn.columns)


def translate_locations(
    locations: pd.DataFrame, x_col: str, y_col: str, type_col: str
) -> pd.DataFrame:
    "Translate locations  to prepare for building connectivity matrix"
    loc_out = pd.DataFrame(index=locations.index, columns=["X", "Y", "Type"])
    loc_out["X"] = locations[x_col] - locations[x_col].min()
    loc_out["Y"] = locations[y_col] - locations[y_col].min()
    loc_out["Type"] = locations[type_col]
    return loc_out


def calc_influence_matrix(locations, y_D, matrix_type="conn", m_max=300):
    assert matrix_type in ["conn", "prod"]
    XA = locations[locations.Type == "Producer"]
    if matrix_type == "prod":
        XB = XA.copy()
    else:
        XB = locations[locations.Type == "Injector"]
    influence_matrix = pd.DataFrame(
        index=pd.MultiIndex.from_product([XA.index, XB.index]), columns=["A"]
    )
    m = 1 + np.arange(m_max)  # elements of sum
    for i, j in influence_matrix.index:
        x_i, y_i = XA.loc[i, ["X", "Y"]]
        x_j, y_j = XB.loc[j, ["X", "Y"]] + 1e-6
        influence_matrix.loc[idx[i, j], "A"] = calc_A_ij(x_i, y_i, x_j, y_j, y_D, m)
    return influence_matrix["A"].unstack()


def calc_A_ij(x_i, y_i, x_j, y_j, y_D, m):
    first_term = (
        2 * np.pi * y_D * (1 / 3.0 - y_i / y_D + (y_i ** 2 + y_j ** 2) / (2 * y_D ** 2))
    )
    return first_term + calc_summed_term(x_i, y_i, x_j, y_j, y_D, m)


def calc_summed_term(x_i, y_i, x_j, y_j, y_D, m):
    "Calculate summed term using Valkó 2000 equations A4-7"
    tm = (
        np.cosh(m * np.pi * (y_D - np.abs(y_i - y_j)))
        + np.cosh(m * np.pi * (y_D - y_i - y_j))
    ) / np.sinh(m * np.pi * y_D)

    S1 = 2 * np.sum(tm / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    tN = tm[-1]
    S2 = -tN / 2 * np.log(
        (1 - np.cos(np.pi * (x_i + x_j))) ** 2 + np.sin(np.pi * (x_i + x_j)) ** 2
    ) - tN / 2 * np.log(
        (1 - np.cos(np.pi * (x_i - x_j))) ** 2 + np.sin(np.pi * (x_i - x_j)) ** 2
    )
    S3 = -2 * tN * np.sum(1 / m * np.cos(m * np.pi * x_i) * np.cos(m * np.pi * x_j))
    return S1 + S2 + S3
