import numpy as np
import pandas as pd
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import jit, vmap

# build influence matrices
idx = pd.IndexSlice


def calc_gains_homogeneous(locations, x_e, y_e):
    locations = locations.copy()
    locations[["X", "Y"]] /= x_e
    y_D = y_e / x_e
    A_prod = calc_influence_matrix(locations, y_D, "prod").astype(float)
    A_conn = calc_influence_matrix(locations, y_D, "conn").astype(float)
    A_prod_inv = jsl.inv(A_prod.values)
    term1 = A_prod_inv / jnp.sum(A_prod_inv)
    term2 =  jnp.ones_like(A_prod_inv) @ A_prod_inv @ A_conn.values - 1
    term3 = A_prod_inv @ A_conn.values
    Lambda = term1 @ term2 - term3
    return pd.DataFrame(Lambda, index=A_prod.index, columns=A_conn.columns)

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
    m = 1 + jnp.arange(m_max)
    for i, j in influence_matrix.index:
        x_i, y_i = XA.loc[i, ["X", "Y"]]
        x_j, y_j = XB.loc[j, ["X", "Y"]] + 1e-6
        influence_matrix.loc[idx[i, j], "A"] = calc_A_ij(x_i, y_i, x_j, y_j, y_D, m)
    return influence_matrix["A"].unstack()


@jit
def calc_A_ij(x_i, y_i, x_j, y_j, y_D, m):
    first_term = (
        2
        * jnp.pi
        * y_D
        * (1 / 3.0 - y_i / y_D + (y_i ** 2 + y_j ** 2) / (2 * y_D ** 2))
    )
    return first_term + calc_summed_term(x_i, y_i, x_j, y_j, y_D, m)


@jit
def calc_summed_term(x_i, y_i, x_j, y_j, y_D, m):
    "Calculate summed term using Valkó 2000 equations A4-7"
    tm = (
        jnp.cosh(m * jnp.pi * (y_D - jnp.abs(y_i - y_j)))
        + jnp.cosh(m * jnp.pi * (y_D - y_i - y_j))
    ) / jnp.sinh(m * jnp.pi * y_D)

    S1 = 2 * jnp.sum(
        tm
        / m
        * jnp.cos(m * jnp.pi * x_i)
        * jnp.cos(m * jnp.pi * x_j)
    )
    tN = tm[-1]
    S2 = -tN / 2 * jnp.log(
        (1 - jnp.cos(jnp.pi * (x_i + x_j))) ** 2 + jnp.sin(jnp.pi * (x_i + x_j)) ** 2
    ) - tN / 2 * jnp.log(
        (1 - jnp.cos(jnp.pi * (x_i - x_j))) ** 2 + jnp.sin(jnp.pi * (x_i - x_j)) ** 2
    )
    S3 = (
        -2 * tN * jnp.sum(1 / m * jnp.cos(m * jnp.pi * x_i) * jnp.cos(m * jnp.pi * x_j))
    )
    return S1 + S2 + S3


