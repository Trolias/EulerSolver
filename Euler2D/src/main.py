import jax
import jax.numpy as jnp
from Update import*
import params
import matplotlib.pyplot as plt
import numpy as np
from helpers import*


# Creating fields
ng = params.gc
x = params.Lxstart + (jnp.arange(params.Nx) + 0.5)*params.dx
y = params.Lystart + (jnp.arange(params.Ny) + 0.5)*params.dy
U = jnp.zeros((params.Ny+2*ng, params.Nx+2*ng, 4))
CSV_L = jnp.zeros((params.Ny+2*ng, params.Nx+2*ng-1, 4))
CSV_R = jnp.zeros_like(CSV_L)
CSV_T = jnp.zeros((params.Ny+2*ng-1, params.Nx+2*ng, 4))
CSV_B = jnp.zeros_like(CSV_T)


# Initialising Fields
rho = 0.0
u = 0.0
p = 0.0

X, Y = jnp.meshgrid(x, y)

if params.cas == 1:
    rho = jnp.where((X <= 0.0) & (Y > 0.0), 0.5323,
          jnp.where((X > 0.0)  & (Y > 0.0), 1.5,
          jnp.where((X <= 0.0) & (Y <= 0.0), 0.138,
                    0.5323))) 

    u = jnp.where((X <= 0.0) & (Y > 0.0), 1.206,
        jnp.where((X > 0.0)  & (Y > 0.0), 0.0,
        jnp.where((X <= 0.0) & (Y <= 0.0), 1.206,
                  0.0)))
    
    v = jnp.where((X <= 0.0) & (Y > 0.0), 0.0,
        jnp.where((X > 0.0)  & (Y > 0.0), 0.0,
        jnp.where((X <= 0.0) & (Y <= 0.0), 1.206,
                  1.206)))


    p = jnp.where((X <= 0.0) & (Y > 0.0), 0.3,
        jnp.where((X > 0.0)  & (Y > 0.0), 1.5,
        jnp.where((X <= 0.0) & (Y <= 0.0), 0.029,
                  0.3)))

elif params.cas == 2:
    rho = jnp.where((X<0.5), 1.0, 0.125)
    u = jnp.where((X<0.5), 0.0, 0.0)
    v = jnp.where((X<0.5), 0.0, 0.0)
    p = jnp.where((X<0.5), 1.0, 0.1)
    U = Boundary_Conditions(U, "Transmissive")
elif params.cas == 3:
    radius = jnp.sqrt(X**2 + Y**2)
    rho = jnp.where(radius < 0.2, 1.0, 0.125)
    u = jnp.where(radius < 0.2, 0.0, 0.0)
    v = jnp.where(radius < 0.2, 0.0, 0.0)
    p = jnp.where(radius < 0.2, 1.0, 0.1)
elif params.cas == 4:
    _rho1 = 2.0
    _rho2 = 1.0
    u0 = 0.5
    p0 = 2.5
    a = 0.02
    delta = 0.05
    sigma = 0.05

    rho = _rho2 + 0.5*(_rho1-_rho2)*(jnp.tanh((Y-0.25)/a) - jnp.tanh((Y-0.75)/a))
    u = u0 * (jnp.tanh((Y-0.25)/a) - jnp.tanh((Y-0.75)/a) - 1)
    v = delta * jnp.sin(2.0*jnp.pi*X) * ( jnp.exp(-((Y-0.25)/sigma)**2) - jnp.exp(-((Y-0.75)/sigma)**2) )
    p = jnp.zeros_like(X) + p0
elif params.cas == 5:
    x0 = 1.0/6.0
    rho = jnp.where((X<x0+Y/jnp.sqrt(3.0)), 8.0, 1.4)
    u = jnp.where((X<x0+Y/jnp.sqrt(3.0)), 8.25*jnp.cos(jnp.pi/3.0), 0.0)
    v = jnp.where((X<x0+Y/jnp.sqrt(3.0)), -8.25*jnp.sin(jnp.pi/3.0), 0.0)
    p = jnp.where((X<x0+Y/jnp.sqrt(3.0)), 116.5, 1.0)
elif params.cas == 6:
    x0 = 5.0
    y0 = 0.0
    beta = 5.0
    r2 = (X-x0)**2 + (Y-y0)**2
    du = -beta/(2.0*jnp.pi)*jnp.exp((1.0-r2)/2)*(Y-y0)
    dv = beta/(2.0*jnp.pi)*jnp.exp((1.0-r2)/2)*(X-x0)
    u = 1.0 + du
    v = 1.0 + dv
    dT = -((params.gamma - 1.0) * beta**2 / (8.0 * params.gamma * jnp.pi**2)) * jnp.exp(1.0 - r2)
    T = 1.0 + dT

    rho = T ** (1.0 / (params.gamma - 1.0))
    p   = rho ** params.gamma


U = U.at[ng:-ng, ng:-ng, 0].set(rho)
U = U.at[ng:-ng, ng:-ng, 1].set(rho*u)
U = U.at[ng:-ng, ng:-ng, 2].set(rho*v)
U = U.at[ng:-ng, ng:-ng, 3].set( p/(params.gamma-1) + 0.5*rho*(u**2 + v**2))


# plt.contourf(X, Y, U[ng:-ng,ng:-ng,0])
PV = calc_PV(U)
# save_tecplot_dat("DMR_init.dat", x, y, PV[ng:-ng,ng:-ng,:])
# plt.show()

t = 0.0
t_plot = params.t_end/4
plot_counter = 0
# save_tecplot_dat(f"Riemann_{plot_counter}.dat", x, y, PV[ng:-ng,ng:-ng,:])
while (t < params.t_end):

    # Update Cell Field
    Uold = U.copy()
    # Calculate Primite Varibaels (PV)
    PV = calc_PV(Uold)
    # Compute suitable time_step based on maximum speed
    dt = calc_dt(Uold[ng:-ng,ng:-ng,:], PV[ng:-ng,ng:-ng,3])
    dt = jnp.minimum(dt, params.t_end - t)
    # jax.debug.print("dt = : {d_t}", d_t = dt)

    # Update time 
    t = t + dt
    # Flux Reconstruction plus flux Calculation through Riemann solvers
    Uold, U_L, U_R, U_T, U_B, Intercell_Flux_x, Intercell_Flux_y = Reconst_plus_Riemann(Uold, CSV_L, CSV_R, CSV_T, CSV_B, t)
    
    # Update Solution
    U_interior = solve(Uold, U_L, U_R, U_T, U_B, Intercell_Flux_x, Intercell_Flux_y, dt, t, params.time_integrators[params.tim])
    U = Uold.at[ng:-ng,ng:-ng,:].set(U_interior)

    if (t >= plot_counter*t_plot):
        PV = calc_PV(U)
        save_tecplot_dat(f"IsentropivVoretx_{plot_counter}.dat", x, y, PV[ng:-ng,ng:-ng,:])
        plot_counter = plot_counter+1


PV = calc_PV(U)
plt.contourf(X, Y, PV[ng:-ng,ng:-ng,0])

plt.show()

# save_tecplot_dat("SodWENOref.dat", x, y, PV[ng:-ng,ng:-ng,:])

# plot_compare("HLLCWENO5001.dat", "Rus_WENO5_Char_Sod101.dat",
#              label_a="Exact", label_b="WENO_char")
