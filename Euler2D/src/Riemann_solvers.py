import jax
import jax.numpy as jnp
import params


def calc_PVD(CVD_faces):
    PVD = jnp.zeros_like(CVD_faces)
    
    rho = jnp.maximum(CVD_faces[:,:, 0], 1e-12)
    u = CVD_faces[:,:, 1] / rho
    v = CVD_faces[:,:, 2] / rho
    E = CVD_faces[:,:, 3]
    p = (params.gamma - 1.0) * (E - 0.5 * rho * (u**2+v**2))
    p   = jnp.maximum(p, 1e-12)
    
    PVD = PVD.at[:,:, 0].set(rho)
    PVD = PVD.at[:,:, 1].set(u)
    PVD = PVD.at[:,:, 2].set(v)
    PVD = PVD.at[:,:, 3].set(p)
    
    return PVD


def calc_flux(CVD_faces, axis=0):
    phi = jnp.zeros_like(CVD_faces)

    rho = jnp.maximum(CVD_faces[:,:,0], 1e-12)
    u = CVD_faces[:, :, 1] / rho
    v = CVD_faces[:, :, 2] / rho
    E = CVD_faces[:, :, 3]
    p = (params.gamma - 1.0) * (E - 0.5 * rho * (u**2+v**2))
    p   = jnp.maximum(p, 1e-12)

    if axis == 0:
        phi = phi.at[:, :, 0].set(rho * u)
        phi = phi.at[:, :, 1].set(rho * u**2 + p)
        phi = phi.at[:, :, 2].set(rho*u*v)
        phi = phi.at[:, :, 3].set(u * (E + p))
    elif axis == 1:
        phi = phi.at[:, :, 0].set(rho * v)
        phi = phi.at[:, :, 1].set(rho*u*v)
        phi = phi.at[:, :, 2].set(rho * v**2 + p)
        phi = phi.at[:, :, 3].set(v * (E + p))

    return phi

def calc_PV(cell):
    rho = jnp.maximum(cell[:,:,0], 1e-12)
    u   = cell[:,:,1] / rho
    v   = cell[:,:,2] / rho
    E   = cell[:,:,3]
    p   = (params.gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    p   = jnp.maximum(p, 1e-12)  

    PV = jnp.stack([rho, u, v, p], axis=-1)
    return PV


def calc_dt(cell, pressure):
    rho = cell[:,:, 0]
    u = cell[:,:, 1] / rho
    v = cell[:,:, 2] / rho
    
    rho_safe = jnp.maximum(rho, 1e-10)
    pressure_safe = jnp.maximum(pressure, 1e-10)
    
    a = jnp.sqrt(params.gamma * pressure_safe / rho_safe)
    max_speed_x = jnp.max(jnp.abs(u) + a)
    max_speed_y = jnp.max(jnp.abs(v) + a)

    max_speed = jnp.maximum(max_speed_x, max_speed_y)
    
    # jax.debug.print("Max sound speed: {a:.6e}", a=jnp.max(a))
    # jax.debug.print("Max wave speed: {speed:.6e}", speed=max_speed)
    
    h = jnp.minimum(params.dx, params.dy)
    dt = params.CFL * h / max_speed
    
    # jax.debug.print("Calculated dt: {dt:.6e}", dt=dt)
    # jax.debug.print("---")
    
    return dt

def calc_sound_directions(PVD_R, PVD_L):

    a_L = jnp.sqrt(params.gamma * PVD_L[:, :, 3] / PVD_L[:, :, 0])
    a_R = jnp.sqrt(params.gamma * PVD_R[:, :, 3] / PVD_R[:, :, 0])

    return a_L, a_R    
    
        
def Riemann_solver(PVD_L, PVD_R, CVD_L, CVD_R, solver = "Rusanov", axis = 0):
    phi_L = calc_flux(CVD_L, axis)
    phi_R = calc_flux(CVD_R, axis)
    a_L, a_R = calc_sound_directions(PVD_R, PVD_L)
    
    if solver == "Rusanov":
        if axis == 0:
            S = jnp.max(jnp.maximum(abs(PVD_R[:, :, 1]) + a_R, abs(PVD_L[:, :, 1]) + a_L))
        elif axis == 1:
            S = jnp.max(jnp.maximum(abs(PVD_R[:, :, 2]) + a_R, abs(PVD_L[:, :, 2]) + a_L))

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
        def calc_Ustar(PVD, U, S, Sstar):
            rho = PVD[:, :, 0]
            u   = PVD[:, :, 1]
            v   = PVD[:, :, 2]
            p   = PVD[:, :, 3]
            E   = U[:, :, 3]

            # normal and tangential velocities
            if axis == 0:       # x-interface
                un, ut = u, v
            else:               # y-interface
                un, ut = v, u

            rho_star = rho * (S - un) / (S - Sstar)

            rho_un_star = rho_star * Sstar
            rho_ut_star = rho_star * ut

            if axis == 0:
                rho_u_star, rho_v_star = rho_un_star, rho_ut_star
            else:
                rho_u_star, rho_v_star = rho_ut_star, rho_un_star

            E_star = rho_star * (
                E / rho + (Sstar - un) * (Sstar + p / (rho * (S - un)))
            )

            return jnp.stack([rho_star, rho_u_star, rho_v_star, E_star], axis=-1)

        unL = PVD_L[:, :, axis + 1]
        unR = PVD_R[:, :, axis + 1]

        S_L = jnp.minimum(unL - a_L, unR - a_R)
        S_R = jnp.maximum(unL + a_L, unR + a_R)

        denom = PVD_L[:, :, 0] * (S_L - unL) - PVD_R[:, :, 0] * (S_R - unR)
        Sstar = (
            (PVD_R[:, :, 3] - PVD_L[:, :, 3]) +
            PVD_L[:, :, 0] * unL * (S_L - unL) -
            PVD_R[:, :, 0] * unR * (S_R - unR)
        ) / denom

        Ustar_L = calc_Ustar(PVD_L, CVD_L, S_L, Sstar)
        Ustar_R = calc_Ustar(PVD_R, CVD_R, S_R, Sstar)

        flux_L_star = phi_L + S_L[:, :, None] * (Ustar_L - CVD_L)
        flux_R_star = phi_R + S_R[:, :, None] * (Ustar_R - CVD_R)

        flux = jnp.where(
            (S_L >= 0.0)[:, :, None], phi_L,
            jnp.where(
                ((S_L <= 0.0) & (Sstar >= 0.0))[:, :, None], flux_L_star,
                jnp.where(
                    (S_R >= 0.0)[:, :, None], flux_R_star,
                    phi_R
                )
            )
        )

        return flux
