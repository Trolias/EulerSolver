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

    def Stencil(U, ax):
        if ax == 0:  # x-direction
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
        else:  # y-direction
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
    
    
    if params.Vars[params.var_counter] == "Characteristic Variables":
        L, R = eigen_LR(var, k_face, axis)  

        WL_stencil = jnp.einsum('lij,slj->sli', L, SL)  # (5,line,4)
        WR_stencil = jnp.einsum('lij,slj->sli', L, SR)  # (5,line,4)

        # Reconstruct in characteristic space
        WL = weno5_local(WL_stencil)  # (line,4)
        WR = weno5_local(WR_stencil)  # (line,4)

        # Back to conservative: U = R W
        UL = jnp.einsum('lij,lj->li', R, WL)  # (line,4)
        UR = jnp.einsum('lij,lj->li', R, WR)  # (line,4)
    else:
        UL = weno5_local(SL)  # (line_length, nvar)
        UR = weno5_local(SR)
    
    return UL, UR

def eigen_LR(U, k_face, axis=0):

    if axis == 0:  # x-direcrtion
        Uavg = 0.5 * (U[:, k_face, :] + U[:, k_face+1, :])  
        nx, ny = 1.0, 0.0
    else:          # y-direction
        Uavg = 0.5 * (U[k_face, :, :] + U[k_face+1, :, :])  # (Nx,4)
        nx, ny = 0.0, 1.0

    tx, ty = -ny, nx  # tangent

    rho = jnp.maximum(Uavg[:, 0], 1e-12)
    u   = Uavg[:, 1] / rho
    v   = Uavg[:, 2] / rho
    E   = Uavg[:, 3]

    p = jnp.maximum((params.gamma - 1.0) * (E - 0.5*rho*(u*u + v*v)), 1e-12)
    a = jnp.sqrt(params.gamma * p / rho)
    H = (E + p) / rho

    un = u*nx + v*ny
    ut = u*tx + v*ty

    a2   = a*a
    gm1  = params.gamma - 1.0
    beta = gm1 / a2
    q2   = un*un + ut*ut

    one  = jnp.ones_like(rho)
    zero = jnp.zeros_like(rho)

    r1 = jnp.stack([one,
                    u - a*nx,
                    v - a*ny,
                    H - a*un], axis=1)

    r2 = jnp.stack([zero,          
                    one*tx,
                    one*ty,
                    ut], axis=1)

    r3 = jnp.stack([one,
                    u,
                    v,
                    0.5*(u*u + v*v)], axis=1)

    r4 = jnp.stack([one,
                    u + a*nx,
                    v + a*ny,
                    H + a*un], axis=1)

    R = jnp.stack([r1, r2, r3, r4], axis=2)  # (line,4,4)

    # ---- Left eigenvectors (rows) analytically ----
    K = 0.25 * beta * q2

    # rows in (rho, m_n, m_t, E)
    l1p = (K + 0.5*un/a,  -0.5*beta*un - 0.5/a,  -0.5*beta*ut,  0.5*beta)
    l2p = (-ut,           0.0*un,               one,          0.0*un)
    l3p = (one - 0.5*beta*q2,  beta*un,          beta*ut,     -beta)
    l4p = (K - 0.5*un/a,  -0.5*beta*un + 0.5/a,  -0.5*beta*ut,  0.5*beta)

    def to_xy(lrho, lmn, lmt, lE):
        lmx = lmn*nx + lmt*tx
        lmy = lmn*ny + lmt*ty
        return jnp.stack([lrho, lmx, lmy, lE], axis=1)  # (line,4)

    l1 = to_xy(*l1p)
    l2 = to_xy(*l2p)
    l3 = to_xy(*l3p)
    l4 = to_xy(*l4p)

    L = jnp.stack([l1, l2, l3, l4], axis=1)  # (line,4,4) with rows

    return L, R







