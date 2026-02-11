import jax
import jax.numpy as jnp
from Update import*
import params
import matplotlib.pyplot as plt
import numpy as np
from helpers import*


# Creating fields
x = jnp.linspace(params.Lxstart, params.Lxend, params.Nx)
y = jnp.linspace(params.Lystart, params.Lyend, params.Ny)
U = jnp.zeros((params.Ny, params.Nx, 4))
CSV_L = jnp.zeros((params.Ny, params.Nx-1, 4))
CSV_R = jnp.zeros_like(CSV_L)
CSV_T = jnp.zeros((params.Ny-1, params.Nx, 4))
CSV_B = jnp.zeros_like(CSV_T)


# Initialising Fields
rho = 0.0
u = 0.0
p = 0.0

X, Y = jnp.meshgrid(x, y)

if params.cas == 1:
    rho = jnp.where((X<=0.0) & (Y>=0.0), 0.5323,
        jnp.where((X>0.0) & (Y>0.0), 1.5,
        jnp.where((X<=0.0) & (Y<0.0), 0.138,
                    0.5323)))
    
    u = jnp.where((X<=0.0) & (Y>=0.0), 1.206,
    jnp.where((X>0.0) & (Y>0.0), 0.0,
    jnp.where((X<=0.0) & (Y<0.0), 1.206,
              0.0)))  

    v = jnp.where((X<=0.0) & (Y>=0.0), 0.0,
        jnp.where((X>0.0) & (Y>0.0), 0.0,
        jnp.where((X<=0.0) & (Y<0.0), 1.206,
                1.206))) 

    p = jnp.where((X<=0.0) & (Y>=0.0), 0.3,
        jnp.where((X>0.0) & (Y>0.0), 1.5,
        jnp.where((X<=0.0) & (Y<0.0), 0.029,
                0.3)))

elif params.cas == 2:
    rho = jnp.where((X<0.5), 1.0, 0.125)
    u = jnp.where((X<0.5), 0.0, 0.0)
    v = jnp.where((X<0.5), 0.0, 0.0)
    p = jnp.where((X<0.5), 1.0, 0.1)
elif params.cas == 3:
    radius = jnp.sqrt(X**2 + Y**2)
    rho = jnp.where(radius < 0.2, 1.0, 0.125)
    u = jnp.where(radius < 0.2, 0.0, 0.0)
    v = jnp.where(radius < 0.2, 0.0, 0.0)
    p = jnp.where(radius < 0.2, 1.0, 0.1)




U = U.at[:, :, 0].set(rho)
U = U.at[:, :, 1].set(rho*u)
U = U.at[:, :, 2].set(rho*v)
U = U.at[:, :, 3].set( p/(params.gamma-1) + 0.5*rho*(u**2 + v**2))

plt.contourf(X, Y, U[:,:,0])
plt.show()

t = 0.0
while (t < params.t_end):

    # Update Cell Field
    Uold = U.copy()
    # Calculate Primite Varibaels (PV)
    PV = calc_PV(Uold)
    # Compute suitable time_step based on maximum speed
    dt = calc_dt(Uold, PV[:,:,3])
    dt = jnp.minimum(dt, params.t_end - t)
    # jax.debug.print("dt = : {d_t}", d_t = dt)

    # Flux Reconstruction plus flux Calculation through Riemann solvers
    Uold, U_L, U_R, U_T, U_B, Intercell_Flux_x, Intercell_Flux_y = Reconst_plus_Riemann(Uold, CSV_L, CSV_R, CSV_T, CSV_B)
    
    # Update Solution
    U_interior = solve(Uold, U_L, U_R, U_T, U_B, Intercell_Flux_x, Intercell_Flux_y, dt, params.time_integrators[params.tim])
    U = U.at[1:-1,1:-1,:].set(U_interior)
    
    # Update time 
    t = t + dt

PV = calc_PV(U)
plt.contourf(X, Y, PV[:,:,0])
# plt.plot(x, PV[params.Ny//2, :, 0])
plt.show()

# plt.plot(x, PV[:,0], color = 'magenta')
# plt.xlim(params.Lstart, params.Lend)
# plt.grid()
# plt.legend()

# plt.show()

save_tecplot_dat("2DRieman801.dat", x, y, PV)

# plot_compare("HLLCWENO5001.dat", "Rus_WENO5_Char_Sod101.dat",
#              label_a="Exact", label_b="WENO_char")
