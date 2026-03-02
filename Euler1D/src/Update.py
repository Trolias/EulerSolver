from jax import jit
from Reconstruction import*
from Riemann_solvers import*
from BCs_Limiters import*
import params   

def solve(CSV, CDL, CDR, flux, dt, intg = "Euler"):

    if intg == "Euler":
        return CSV[1:-1,:] - dt/params.dx * (flux[1:,:] - flux[:-1,:]) 
    

    elif intg == "RK3_TVD":
        U1 = jnp.zeros_like(CSV)  
        U1 = U1.at[1:-1,:].set(CSV[1:-1, :] - dt/params.dx * (flux[1:,:] - flux[:-1, :]))

        U2 = jnp.zeros_like(CSV)
        U2, CDL2, CDR2, f1 = Reconst_plus_Riemann(U1, CDL, CDR) 
        U2 = U2.at[1:-1, :].set( 0.75 * CSV[1:-1, :] + 0.25 * U1[1:-1,:] - (0.25*dt/params.dx) * (f1[1:,:] - f1[:-1,:]) )
    
        _, _, _, f2 = Reconst_plus_Riemann(U2, CDL2, CDR2) 
    
        return 1.0/3.0*CSV[1:-1,:] + 2.0/3.0*U2[1:-1,:] - (2.0/3.0*dt/params.dx) * (f2[1:,:] - f2[:-1,:])
    

@jit
def Reconst_plus_Riemann(cell, U_L, U_R):

    U_L, U_R = fluxReconstruction(cell, U_L, U_R, params.Scheme[params.shm], params.limiters[params.lim])
    Uold, U_L, U_R = Boundary_Conditions(cell, U_L, U_R)

    # jax.debug.print("U_L min/max: {mn} {mx}", mn=jnp.min(U_L), mx=jnp.max(U_L))
    # jax.debug.print("U_R min/max: {mn} {mx}", mn=jnp.min(U_R), mx=jnp.max(U_R))

    # jax.debug.print("U_L zeros: {z}", z=jnp.sum(jnp.isclose(U_L, 0.0)))
    # jax.debug.print("U_R zeros: {z}", z=jnp.sum(jnp.isclose(U_R, 0.0)))


    PV_L = calc_PVD(U_L)
    PV_R = calc_PVD(U_R)
    # Intercell_Flux = Riemann_solver(PV_L[:-1,:], PV_R[1:,:], U_L[:-1,:], U_R[1:,:], params.Riemann_solvers[params.riem_sol])
    Intercell_Flux = Riemann_solver(PV_L, PV_R, U_L, U_R, params.Riemann_solvers[params.riem_sol])

    
    return Uold, U_L, U_R, Intercell_Flux