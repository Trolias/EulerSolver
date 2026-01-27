import jax
import jax.numpy as jnp
from Update import*
import params
import matplotlib.pyplot as plt
import numpy as np
from helpers import*


# Creating fields
x = jnp.linspace(params.Lstart, params.Lend, params.N)
U = jnp.zeros((params.N,3))
CSV_L = jnp.zeros((params.N-1, 3))
CSV_R = jnp.zeros_like(CSV_L)

# Initialising Fields
rho = 0.0
u = 0.0
p = 0.0

for i in range(params.N):
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
            rho = 1.0 + 0.5*jnp.sin(5.0*x[i])
            u = 0.0
            p = 1.0

    U = U.at[i, 0].set(rho)
    U = U.at[i, 1].set(rho*u)
    U = U.at[i, 2].set( p/(params.gamma-1) + 0.5*rho*u**2)


t = 0.0
while (t < params.t_end):

    # Update Cell Field
    Uold = U.copy()
    # Calculate Primite Varibaels (PV)
    PV = calc_PV(Uold)
    # Compute suitable time_step based on maximum speed
    dt = calc_dt(Uold, PV[:,2])
    dt = jnp.minimum(dt, params.t_end - t)
    # jax.debug.print("dt = : {d_t}", d_t = dt)

    # Flux Reconstruction plus flux Calculation through Riemann solvers
    Uold, U_L, U_R, Intercell_Flux = Reconst_plus_Riemann(Uold, CSV_L, CSV_R)
    
    # Update Solution
    U_interior = solve(Uold, U_L, U_R, Intercell_Flux, dt, params.time_integrators[params.tim])
    U = U.at[1:-1,:].set(U_interior)
    
    # Update time 
    t = t + dt

PV = calc_PV(U)

plt.plot(x, PV[:,0], color = 'magenta')
plt.xlim(params.Lstart, params.Lend)
plt.grid()
plt.legend()

plt.show()

save_dat("HLLC_MUSCL_ShuOsher801.dat", x, PV)

plot_compare("HLLC_MUSCL_ShuOsher801.dat", "HLLC_WENO5_Char_ShuOsher801.dat",
             label_a="WENO_cons", label_b="WENO_char")
