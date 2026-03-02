import jax
import jax.numpy as jnp
from Update import*
import params
import matplotlib.pyplot as plt
import numpy as np
from helpers import*


# Creating fields
ng = params.gc # Number of ghost nodes
x = params.Lstart + (jnp.arange(params.N) + 0.5)*params.dx
U = jnp.zeros((params.N + 2*ng ,3))
CSV_L = jnp.zeros((params.N + 2*ng-1, 3))
CSV_R = jnp.zeros_like(CSV_L)

# Initialising Fields
rho = 0.0
u = 0.0
p = 0.0

for i in range(params.N+ng):
    if (params.cas == 1):

        if x[i] <= 0.5:  
            rho = 1.0
            u =  0.0
            p = 1.0
        else:
            rho = 0.125
            u = 0.0
            p = 0.1

    elif (params.cas == 2):

        if x[i] < -4.0:   
            rho = 3.857143
            u =  2.629369
            p = 10.3333
        else:
            rho = 1.0 + 0.2*jnp.sin(5.0*x[i])
            u = 0.0
            p = 1.0

    U = U.at[i, 0].set(rho)
    U = U.at[i, 1].set(rho*u)
    U = U.at[i, 2].set( p/(params.gamma-1) + 0.5*rho*u**2)


U, CSV_L, CSV_R = Boundary_Conditions(U, CSV_L, CSV_R)    # !!! This line is IMPORTANT !!!
# PV = calc_PV(U[ng:-ng,:])
# plt.plot(x,PV[:,0])
# plt.show()


t = 0.0
while (t < params.t_end):

    # Update Cell Field
    Uold = U.copy()
    # Calculate Primite Varibaels (PV)
    PV = calc_PV(Uold)
    # Compute suitable time_step based on maximum speed
    dt = calc_dt(Uold[ng:-ng], PV[ng:-ng,2])
    dt = jnp.minimum(dt, params.t_end - t)
    # jax.debug.print("dt = : {d_t}", d_t = dt)

    # Flux Reconstruction plus flux Calculation through Riemann solvers
    Uold, U_L, U_R, Intercell_Flux = Reconst_plus_Riemann(Uold, CSV_L, CSV_R)
    
    # Update Solution
    U_interior = solve(Uold, U_L, U_R, Intercell_Flux, dt, params.time_integrators[params.tim])
    U = Uold.at[ng:-ng,:].set(U_interior)
    
    # Update time 
    t = t + dt

PV = calc_PV(U[ng:-ng,:])

plt.plot(x, PV[:,0], color = 'magenta')
plt.xlim(params.Lstart, params.Lend)
plt.grid()
plt.legend()

plt.show()

save_dat("DebugWENO.dat", x, PV)

# plot_compare("DebugWENO.dat","ShuOsher5001.dat",
#              label_a="Debug", label_b="Exact")
