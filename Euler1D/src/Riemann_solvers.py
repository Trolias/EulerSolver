import jax
import jax.numpy as jnp
import params


def calc_PVD(CVD_faces):
    PVD = jnp.zeros_like(CVD_faces)
    
    rho = jnp.maximum(CVD_faces[:, 0], 1e-12)
    u = CVD_faces[:, 1] / rho
    E = CVD_faces[:, 2]
    p = (params.gamma - 1.0) * (E - 0.5 * rho * u**2)
    p   = jnp.maximum(p, 1e-12)
    
    PVD = PVD.at[:, 0].set(rho)
    PVD = PVD.at[:, 1].set(u)
    PVD = PVD.at[:, 2].set(p)
    
    return PVD


def calc_flux(CVD_faces):
    phi = jnp.zeros_like(CVD_faces)

    rho = CVD_faces[:, 0]
    rho = jnp.maximum(CVD_faces[:,0], 1e-12)
    u = CVD_faces[:, 1] / rho
    E = CVD_faces[:, 2]
    p = (params.gamma - 1.0) * (E - 0.5 * rho * u**2)
    p   = jnp.maximum(p, 1e-12)


    phi = phi.at[:, 0].set(rho * u)
    phi = phi.at[:, 1].set(p + rho * u**2)
    phi = phi.at[:, 2].set(u * (E + p))

    return phi

def calc_PV(cell):
    rho = jnp.maximum(cell[:,0], 1e-12)
    u   = cell[:,1] / rho
    E   = cell[:,2]
    p   = (params.gamma - 1.0) * (E - 0.5 * rho * u**2)
    p   = jnp.maximum(p, 1e-12)  

    PV = jnp.stack([rho, u, p], axis=1)
    return PV


def calc_dt(cell, pressure):
    # Add safety checks
    rho = cell[:, 0]
    u = cell[:, 1] / rho
    
    # Debug prints that work with JAX JIT
    # jax.debug.print("Min/Max density: {min_rho:.6e}, {max_rho:.6e}", 
    #             min_rho=jnp.min(rho), max_rho=jnp.max(rho))
    # jax.debug.print("Min/Max pressure: {min_p:.6e}, {max_p:.6e}", 
    #             min_p=jnp.min(pressure), max_p=jnp.max(pressure))
    # jax.debug.print("Min/Max velocity: {min_u:.6e}, {max_u:.6e}", 
    #             min_u=jnp.min(u), max_u=jnp.max(u))
    
    # # Check for negatives using jax.lax.cond for warnings
    # jax.debug.print("Negative density: {neg}", neg=jnp.any(rho <= 0))
    # jax.debug.print("Negative pressure: {neg}", neg=jnp.any(pressure <= 0))
    
    # Safeguard against negative values
    rho_safe = jnp.maximum(rho, 1e-10)
    pressure_safe = jnp.maximum(pressure, 1e-10)
    
    a = jnp.sqrt(params.gamma * pressure_safe / rho_safe)
    max_speed = jnp.max(jnp.abs(u) + a)
    
    # jax.debug.print("Max sound speed: {a:.6e}", a=jnp.max(a))
    # jax.debug.print("Max wave speed: {speed:.6e}", speed=max_speed)
    
    dt = params.CFL * params.dx / max_speed
    
    # jax.debug.print("Calculated dt: {dt:.6e}", dt=dt)
    # jax.debug.print("---")
    
    return dt

def calc_sound_directions(PVD_R, PVD_L):

    a_L = jnp.sqrt(params.gamma * PVD_L[:, 2] / PVD_L[:, 0])
    a_R = jnp.sqrt(params.gamma * PVD_R[:, 2] / PVD_R[:, 0])

    return a_L, a_R    
    
        
def Riemann_solver(PVD_L, PVD_R, CVD_L, CVD_R, solver = "Rusanov"):
    phi_L = calc_flux(CVD_L)
    phi_R = calc_flux(CVD_R)
    a_L, a_R = calc_sound_directions(PVD_R, PVD_L)
    
    if solver == "Rusanov":
        S = jnp.max(jnp.maximum(abs(PVD_R[:, 1]) + a_R, abs(PVD_L[:, 1]) + a_L))
        return 0.5 * (phi_L + phi_R) - 0.5*S * (CVD_R - CVD_L) 
    
    elif solver == "HLL":
        S_L = jnp.minimum(PVD_L[:, 1] - a_L, PVD_R[:, 1] - a_R)
        S_R = jnp.maximum(PVD_L[:, 1] + a_L, PVD_R[:, 1] + a_R)
        
        denom = jnp.maximum(S_R - S_L, 1e-12)
        
        # HLL flux 
        flux_hll = (S_R[:, None]*phi_L - S_L[:, None]*phi_R + 
                    S_L[:, None]*S_R[:, None] * (CVD_R - CVD_L)) / denom[:, None]
        
        flux = jnp.where((S_L >= 0.0)[:, None], phi_L, flux_hll)
        flux = jnp.where((S_R <= 0.0)[:, None], phi_R, flux)

        return flux

    elif solver == "HLLC":
        def calc_Ustar(PVD, CVD, S, Sstar):
            rho = PVD[:, 0]
            u = PVD[:, 1]
            p = PVD[:, 2]
            E = CVD[:, 2]
            
            factor = rho * (S - u) / (S - Sstar)
            
            rho_star = factor
            rho_u_star = factor * Sstar
            E_star = factor * (E/rho + (Sstar - u) * (Sstar + p/(rho * (S - u))))
            
            return jnp.stack([rho_star, rho_u_star, E_star], axis=1)
        
        S_L = jnp.minimum(PVD_L[:, 1] - a_L, PVD_R[:, 1] - a_R)
        S_R = jnp.maximum(PVD_L[:, 1] + a_L, PVD_R[:, 1] + a_R)
        
        Sstar = (
            PVD_R[:, 2] - PVD_L[:, 2] + 
            PVD_L[:, 0] * PVD_L[:, 1] * (S_L - PVD_L[:, 1]) - 
            PVD_R[:, 0] * PVD_R[:, 1] * (S_R - PVD_R[:, 1])
        ) / (
            PVD_L[:, 0] * (S_L - PVD_L[:, 1]) - 
            PVD_R[:, 0] * (S_R - PVD_R[:, 1])
        )
        
        Ustar_L = calc_Ustar(PVD_L, CVD_L, S_L, Sstar)
        Ustar_R = calc_Ustar(PVD_R, CVD_R, S_R, Sstar)
        
        flux_L_star = phi_L + S_L[:, None] * (Ustar_L - CVD_L)
        flux_R_star = phi_R + S_R[:, None] * (Ustar_R - CVD_R)
        
        flux = jnp.where((S_L[:, None] >= 0.0), phi_L,
                jnp.where((Sstar[:, None] >= 0.0) & (S_L[:, None] <= 0.0), flux_L_star,
                 jnp.where((S_R[:, None] >= 0.0), flux_R_star, phi_R)))
        
        return flux