import jax 
import jax.numpy as jnp
import params

def limiter(cell, name = "Minmod"):
    # Define smoothness_indicator " $r_{\pm{\frac{1}{2}}}$ "
    r_plus_half = (cell[2:, :] - cell[1:-1, :]) / (cell[1:-1, :] - cell[:-2, :] + 1e-08)
    r_minus_half = (cell[1:-1, :] - cell[:-2, :]) / (cell[2:, :] - cell[1:-1, :] + 1e-08)

    psi_r = jnp.maximum(0.0, jnp.minimum(1.0, r_plus_half))
    psi_l = jnp.maximum(0.0, jnp.minimum(1.0, r_minus_half))

    return psi_l, psi_r


def Boundary_Conditions(cell, CDL, CDR, name = "Transmissive", ng = params.gc):

    if name == "Transmissive":
        cell = cell.at[:ng, :].set(cell[ng, :])
        cell = cell.at[-ng:, :].set(cell[-ng-1, :])

    return cell, CDL, CDR