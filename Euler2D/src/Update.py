from jax import jit
from Reconstruction import*
from Riemann_solvers import*
from BCs_Limiters import*
import params   

def solve(CSV, CDL, CDR, CDT, CDB, flux_x, flux_y, dt, intg = "Euler"):

    if intg == "Euler":
        return CSV[1:-1,1:-1,:] - dt * ( (flux_x[1:-1,1:,:] - flux_x[1:-1,:-1,:])/params.dx + (flux_y[1:,1:-1,:] - flux_y[:-1,1:-1,:])/params.dy) 
    

    elif intg == "RK3_TVD":
        U1 = CSV 
        U1 = U1.at[1:-1,1:-1,:].set(CSV[1:-1,1:-1, :] - dt * ( 
            + (flux_x[1:-1, 1:, :] - flux_x[1:-1, :-1, :])/params.dx
            + (flux_y[1:, 1:-1, :] - flux_y[:-1, 1:-1, :])/params.dy
        ))

        U2 = U1
        U2, CDL2, CDR2, CDT2, CDB2, f1x, f1y = Reconst_plus_Riemann(U1, CDL, CDR, CDT, CDB) 
        U2 = U2.at[1:-1,1:-1, :].set( 0.75 * CSV[1:-1,1:-1, :] 
                    + 0.25 * U1[1:-1,1:-1,:] 
                    - 0.25*dt * (
                    + (f1x[1:-1, 1:, :] - f1x[1:-1, :-1, :])/params.dx
                    + (f1y[1:, 1:-1, :] - f1y[:-1, 1:-1, :])/params.dy    
                    ))
    
        _, _, _, _, _, f2x, f2y = Reconst_plus_Riemann(U2, CDL2, CDR2, CDT2, CDB2)
        U3 = U2.at[1:-1, 1:-1, :].set(
                1.0/3.0 * CSV[1:-1, 1:-1, :] + 
                2.0/3.0 * U2[1:-1, 1:-1, :] - 
                2.0/3.0 * dt * (
                    (f2x[1:-1, 1:, :] - f2x[1:-1, :-1, :])/params.dx +
                    (f2y[1:, 1:-1, :] - f2y[:-1, 1:-1, :])/params.dy
                )
            )
            
        return U3[1:-1, 1:-1, :]
    

@jit
def Reconst_plus_Riemann(cell, U_L, U_R, U_T, U_B):

    U_L, U_R, U_T, U_B = fluxReconstruction(cell, U_L, U_R, U_T, U_B, params.Scheme[params.shm], params.limiters[params.lim])
    Uold, U_L, U_R, U_T, U_B = Boundary_Conditions(cell, U_L, U_R, U_T, U_B)

    # jax.debug.print("U_L min/max: {mn} {mx}", mn=jnp.min(U_L), mx=jnp.max(U_L))
    # jax.debug.print("U_R min/max: {mn} {mx}", mn=jnp.min(U_R), mx=jnp.max(U_R))

    # jax.debug.print("U_L zeros: {z}", z=jnp.sum(jnp.isclose(U_L, 0.0)))
    # jax.debug.print("U_R zeros: {z}", z=jnp.sum(jnp.isclose(U_R, 0.0)))


    PV_L = calc_PVD(U_L)
    PV_R = calc_PVD(U_R)
    PV_T = calc_PVD(U_T)
    PV_B = calc_PVD(U_B)
    # Intercell_Flux = Riemann_solver(PV_L[:-1,:], PV_R[1:,:], U_L[:-1,:], U_R[1:,:], params.Riemann_solvers[params.riem_sol])
    Intercell_Flux_x = Riemann_solver(PV_L, PV_R, U_L, U_R, params.Riemann_solvers[params.riem_sol], axis=0)
    Intercell_Flux_y = Riemann_solver(PV_T, PV_B, U_T, U_B, params.Riemann_solvers[params.riem_sol], axis=1)

    
    return Uold, U_L, U_R, U_T, U_B, Intercell_Flux_x, Intercell_Flux_y