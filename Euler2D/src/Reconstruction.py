import jax 
import jax.numpy as jnp
from jax import jacobian
from Riemann_solvers import calc_flux, calc_PV, calc_PVD, calc_sound_directions
from BCs_Limiters import*
import params
from jax import vmap

                      
def fluxReconstruction(cell, CDL, CDR, CDT, CDB, scheme = "Constant", limiter_type = "Minmod"):
    if scheme == "Constant":

        CDL = cell[:, :-1,:]
        CDR = cell[:, 1:,:]
        CDT = cell[:-1, :,:]
        CDB = cell[1:, :,:]

    elif scheme == "MUSCL":
        psi_l, psi_r, psi_t, psi_b = limiter(cell, limiter_type)
        # X-Dimension
        CDL = CDL.at[:,1:,:].set(cell[:, 1:-1,:] + 0.5 * psi_r * (cell[:, 1:-1, :] - cell[:, :-2, :]))
        CDR = CDR.at[:,:-1,:].set(cell[:, 1:-1,:] - 0.5 * psi_l * (cell[:, 2:, :] - cell[:, 1:-1, :]))
        # Y-Dimension
        CDT = CDT.at[1:,:,:].set(cell[1:-1, :,:] + 0.5 * psi_t * (cell[1:-1, :, :] - cell[:-2, :, :]))
        CDB = CDB.at[:-1,:,:].set(cell[1:-1, :,:] - 0.5 * psi_b * (cell[2:, :, :] - cell[1:-1, :, :]))

    elif scheme == "WENO":

        CDL, CDR = WENO5(cell, axis=0)
        CDT, CDB = WENO5(cell, axis=1)
                                                                   
    return CDL, CDR, CDT, CDB
                                                                                        


def weno5_local(stencil_5, eps=1e-12):
    # stencil_5: (5, line, nvar) or (5, line)
    v0, v1, v2, v3, v4 = stencil_5[0], stencil_5[1], stencil_5[2], stencil_5[3], stencil_5[4]

    beta0 = (1.0/3.0) * (4*v0*v0 - 19*v0*v1 + 25*v1*v1 + 11*v0*v2 - 31*v1*v2 + 10.0*v2*v2)
    beta1 = (1.0/3.0) * (4*v1*v1 - 13*v1*v2 + 13*v2*v2 +  5*v1*v3 - 13*v2*v3 + 4*v3*v3)
    beta2 = (1.0/3.0) * (10*v2*v2 - 31*v2*v3 + 25*v3*v3 + 11*v2*v4 - 19*v3*v4 + 4*v4*v4)

    g0, g1, g2 = 1.0/16.0, 5.0/8.0, 5.0/16.0

    a0 = g0 / (eps + beta0)**2
    a1 = g1 / (eps + beta1)**2
    a2 = g2 / (eps + beta2)**2
    asum = a0 + a1 + a2

    w0, w1, w2 = a0/asum, a1/asum, a2/asum

    p0 = 3.0/8.0*v0 - 5.0/4.0*v1 + 15.0/8.0*v2
    p1 = -1.0/8.0*v1 + 3.0/4.0*v2 + 3.0/8.0*v3
    p2 = 3.0/8.0*v2 + 3.0/4.0*v3 - 1.0/8.0*v4

    return w0*p0 + w1*p1 + w2*p2


def WENO5(U, axis=0, vars=params.Vars[params.var_counter], eps=1e-6):
    Ny, Nx, nvar = U.shape
    eps = jnp.asarray(eps, dtype=U.dtype)

    if axis == 0:  # x-interfaces
        N = Nx
        U_L = U[:, :-1, :]
        U_R = U[:,  1:, :]
    else:          # y-interfaces
        N = Ny
        U_L = U[:-1, :, :]
        U_R = U[ 1:, :, :]

    idx_x = jnp.arange(2, N - 3)  # 2..N-4

    vmapped_interface = jax.vmap(interface, in_axes=(None, 0, None))
    ULs, URs = vmapped_interface(U, idx_x, axis) 

    if axis == 0:
        # ULs: (nfaces, Ny, nvar) -> (Ny, nfaces, nvar)
        ULs = jnp.swapaxes(ULs, 0, 1)
        URs = jnp.swapaxes(URs, 0, 1)
        U_L = U_L.at[:, idx_x, :].set(ULs)
        U_R = U_R.at[:, idx_x, :].set(URs)
    else:
        # ULs: (nfaces, Nx, nvar) matches U_L[idx_x, :, :]
        U_L = U_L.at[idx_x, :, :].set(ULs)
        U_R = U_R.at[idx_x, :, :].set(URs)

    return U_L, U_R


def interface(var, k_face, axis=0):
    """
    var: (Ny, Nx, nvar) for axis=0, or can be transposed
    k_face: interface index
    axis: 0 for x-direction, 1 for y-direction
    """
    def Stencil(U, ax):
        if ax == 0:  # x-direction: slice along axis=1 (columns)
            SL = jnp.stack([U[:, k_face-2, :],
                            U[:, k_face-1, :],
                            U[:, k_face,   :],
                            U[:, k_face+1, :],
                            U[:, k_face+2, :]], axis=0)  # (5, Ny, nvar)
            
            SR = jnp.stack([U[:, k_face+3, :],
                            U[:, k_face+2, :],
                            U[:, k_face+1, :],
                            U[:, k_face,   :],
                            U[:, k_face-1, :]], axis=0)
        else:  # y-direction: slice along axis=0 (rows)
            SL = jnp.stack([U[k_face-2, :, :],
                            U[k_face-1, :, :],
                            U[k_face,   :, :],
                            U[k_face+1, :, :],
                            U[k_face+2, :, :]], axis=0)  # (5, Nx, nvar)
            
            SR = jnp.stack([U[k_face+3, :, :],
                            U[k_face+2, :, :],
                            U[k_face+1, :, :],
                            U[k_face,   :, :],
                            U[k_face-1, :, :]], axis=0)
        return SL, SR

    SL, SR = Stencil(var, axis)
    
    # SL and SR now have shape (5, line_length, nvar)
    # where line_length = Ny for axis=0, Nx for axis=1
    
    if params.Vars[params.var_counter] == "Characteristic Variables":
        L, R = eigen_LR(var, k_face, axis)  # Also needs axis!
        # ... rest of characteristic variable logic
        WL_stencil = jnp.einsum('ij,sjk->sik', L, SL)  # Better than (L @ SL.T).T
        WR_stencil = jnp.einsum('ij,sjk->sik', R, SR)
        
        WL = weno5_local(WL_stencil)
        UL = R @ WL
        WR = weno5_local(WR_stencil)
        UR = R @ WR
    else:
        UL = weno5_local(SL)  # (line_length, nvar)
        UR = weno5_local(SR)
    
    return UL, UR

def eigen_LR(U, i):

    Uavg = 0.5*(U[i] + U[i+1]) 

    rho = jnp.maximum(Uavg[0], 1e-12)
    u = Uavg[1] / rho
    E = Uavg[2]
    p = jnp.maximum((params.gamma - 1.0) * (E - 0.5 * rho * u*u), 1e-12)

    a = jnp.sqrt(params.gamma * p / rho)
    H = (E + p) / rho

    # Eigenvalue Matrix ' R '
    r1 = jnp.array([1.0, u - a, H - u*a])
    r2 = jnp.array([1.0, u,     0.5*u*u])
    r3 = jnp.array([1.0, u + a, H + u*a])
    R  = jnp.stack([r1, r2, r3], axis=1)

    # Left eigenvectors ' L = R_inverse '
    L0 = jnp.array([0.25*(params.gamma-1.0)*u*u/(a**2) + 0.5*u/a,
                    -0.5*(params.gamma-1.0)*u/(a**2) - 0.5/a,
                    0.5*(params.gamma-1.0)/(a**2)])
    L1 = jnp.array([1.0 - 0.5*(params.gamma-1.0)*u*u/(a**2),
                    (params.gamma-1.0)*u/(a**2),
                    -(params.gamma-1.0)/(a**2)])
    L2 = jnp.array([0.25*(params.gamma-1.0)*u*u/(a**2) - 0.5*u/a,
                    -0.5*(params.gamma-1.0)*u/(a**2) + 0.5/a,
                    0.5*(params.gamma-1.0)/(a**2)])
    L  = jnp.stack([L0, L1, L2], axis=0)  

    return L, R




