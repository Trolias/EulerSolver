import jax 
import jax.numpy as jnp
import params

def limiter(cell, name = "Van Leer"):
    # Define smoothness_indicator " $r_{\pm{\frac{1}{2}}}$ "
    r_plus_half_x = (cell[:, 2:, :] - cell[:, 1:-1, :]) / (cell[:, 1:-1, :] - cell[:, :-2, :] + 1e-08)
    r_minus_half_x = (cell[:, 1:-1, :] - cell[:, :-2, :]) / (cell[:, 2:, :] - cell[:, 1:-1, :] + 1e-08)

    r_plus_half_y = (cell[2:, :, :] - cell[1:-1, :, :]) / (cell[1:-1, :, :] - cell[:-2, :, :] + 1e-08)
    r_minus_half_y = (cell[1:-1, :, :] - cell[:-2, :, :]) / (cell[2:, :, :] - cell[1:-1, :, :] + 1e-08)
    if name == 'Minmod':
        psi_r = jnp.maximum(0.0, jnp.minimum(1.0, r_plus_half_x))
        psi_l = jnp.maximum(0.0, jnp.minimum(1.0, r_minus_half_x))
        psi_t = jnp.maximum(0.0, jnp.minimum(1.0, r_plus_half_y))
        psi_b = jnp.maximum(0.0, jnp.minimum(1.0, r_minus_half_y))
    elif name == "Van Leer":
        psi_r = (r_plus_half_x + abs(r_plus_half_x))/(1.0 + abs(r_plus_half_x))
        psi_l = (r_minus_half_x + abs(r_minus_half_x))/(1.0 + abs(r_minus_half_x))
        psi_t = (r_plus_half_y + abs(r_plus_half_y))/(1.0 + abs(r_plus_half_y))
        psi_b = (r_minus_half_y + abs(r_minus_half_y))/(1.0 + abs(r_minus_half_y))

        # psi_r = jnp.maximum(0.0, jnp.minimum(jnp.minimum(0.5*(1.0+r_plus_half_x), 2.0*r_plus_half_x), 2))
        # psi_l = jnp.maximum(0.0, jnp.minimum(jnp.minimum(0.5*(1.0+r_minus_half_x), 2.0*r_minus_half_x), 2))
        # psi_t = jnp.maximum(0.0, jnp.minimum(jnp.minimum(0.5*(1.0+r_plus_half_y), 2.0*r_plus_half_y), 2))
        # psi_b = jnp.maximum(0.0, jnp.minimum(jnp.minimum(0.5*(1.0+r_minus_half_y), 2.0*r_minus_half_y), 2))


    return psi_l, psi_r, psi_t, psi_b


def Boundary_Conditions(cell, t, name = "Periodic", ng = params.gc, ):

    if name == "Transmissive":
        cell = cell.at[:, :ng, :].set(cell[:, ng:ng+1, :])
        cell = cell.at[:, -ng:, :].set(cell[:, -ng-1:-ng, :])
        cell = cell.at[:ng, :, :].set(cell[ng:ng+1, :, :])
        cell = cell.at[-ng:, :, :].set(cell[-ng-1:-ng, :, :])
    elif name == "Periodic":
        # --- X-Direction Periodicity ---
        cell = cell.at[0:ng, :, :].set(cell[-2*ng:-ng, :, :])
        cell = cell.at[-ng:, :, :].set(cell[ng:2*ng, :, :])

        # --- Y-Direction Periodicity ---
        cell = cell.at[:, 0:ng, :].set(cell[:, -2*ng:-ng, :])
        cell = cell.at[:, -ng:, :].set(cell[:, ng:2*ng, :])
    elif name == "DM reflection":
        x = params.Lxstart + (jnp.arange(params.Nx)+0.5)*params.dx
        y = params.Lystart + (jnp.arange(params.Ny)+0.5)*params.dy
        x0 = 1.0/6.0
        v_shock = 10.0
        _rho0 = 1.4
        _u0 = 0.0
        _v0 = 0.0
        _p0 = 1.0
        _rho1 = 8.0
        _u1 = 8.25*jnp.cos(jnp.pi/3.0)
        _v1 = -8.25*jnp.sin(jnp.pi/3.0)
        _p1 = 116.5

        def set_state(rho, u, v, p):
            E = p/(params.gamma-1.0) + 0.5*rho*(u*u + v*v)
            return jnp.array([rho, rho*u, rho*v, E])
        pre_shock = set_state(_rho0, _u0, _v0, _p0)
        post_shock = set_state(_rho1, _u1, _v1, _p1)

        # BOTTOM
        interior = cell[ng:2*ng, ng:-ng, :]                    # (ng, Nx, 4)
        refl_bottom = interior[::-1, :, :].copy()
        refl_bottom = refl_bottom.at[:, :, 2].set(-refl_bottom[:, :, 2])
        post_shock_bottom = jnp.broadcast_to(post_shock, (ng, params.Nx, 4))
        x_switch = x0 + v_shock * t
        mask_x = (x < x_switch)[None, :, None]
        cell = cell.at[:ng, ng:-ng, :].set(jnp.where(mask_x, post_shock_bottom, refl_bottom))

        # TOP boundary - shock position moves with time
        shock_x = x0 + 1.0/jnp.sqrt(3.0) + v_shock * t
        mask_x = (x < shock_x)[None, :, None]   # (1, Nx, 1)
        cell = cell.at[-ng:, ng:-ng, :].set(
            jnp.where(mask_x, post_shock, pre_shock)
        )

        # LEFT (time-dependent shock)
        y_cut = jnp.sqrt(3.0) * (x0 - v_shock * t) - jnp.sqrt(3.0)/6.0   # correct

        # post-shock if y > y_cut
        mask_y = (y > y_cut)[:, None]                         # (Ny,1)
        cell_vals = jnp.where(mask_y, post_shock, pre_shock)  # (Ny,4)
        cell = cell.at[ng:-ng, :ng, :].set(cell_vals[:, None, :])


        # RIGHT (outflow)
        last = cell[:, -ng-1, :][:, None, :]                   # (Ny+2ng, 1, 4)
        cell = cell.at[:, -ng:, :].set(last)


    return cell
