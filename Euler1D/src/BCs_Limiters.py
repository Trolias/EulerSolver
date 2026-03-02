import jax 
import jax.numpy as jnp

def limiter(cell, name = "Minmod"):
    # Define smoothness_indicator " $r_{\pm{\frac{1}{2}}}$ "
    r_plus_half = (cell[2:, :] - cell[1:-1, :]) / (cell[1:-1, :] - cell[:-2, :] + 1e-08)
    r_minus_half = (cell[1:-1, :] - cell[:-2, :]) / (cell[2:, :] - cell[1:-1, :] + 1e-08)

    psi_r = jnp.maximum(0.0, jnp.minimum(1.0, r_plus_half))
    psi_l = jnp.maximum(0.0, jnp.minimum(1.0, r_minus_half))

    return psi_l, psi_r


def Boundary_Conditions(cell, CDL, CDR, name = "Constant"):

    if name == "Constant":
        cell = cell.at[0, :].set(cell[1, :])
        cell = cell.at[-1, :].set(cell[-2, :])
    
        CDL = CDL.at[0,:].set(cell[0,:])
        CDR = CDR.at[0,:].set(cell[1,:])

        CDL = CDL.at[-1,:].set(cell[-2,:])
        CDR = CDR.at[-1,:].set(cell[-1,:])

    return cell, CDL, CDR