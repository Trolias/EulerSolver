N =801
gamma = 1.4
CFL = 0.3

tim = 2
time_integrators = {
    1: "Euler",
    2: "RK3_TVD"
}

lim = 1
limiters = {
    1: 'Minmod',
    2: 'VanLeer',
}

shm = 2
Scheme = {
    1: 'Constant',
    2: 'MUSCL',
    3: 'WENO',
}

var_counter = 1
Vars = {
    1: "Conserved Variables",
    2: "Characteristic Variables"
}

riem_sol = 3
Riemann_solvers = {
    1: 'Rusanov',
    2: 'HLL',
    3: 'HLLC',
}

cas = 2
case = {
    1: "Sod",
    2: "Shu-Osher"
}

if cas == 1:    # Sod's Shock Tube
    Lstart = 0.0
    Lend = 1.0
    t_end = 0.2
elif cas == 2:  # Shu_Osher
    Lstart = -5.0  
    Lend = 5.0
    t_end = 1.8
else:
    L = 1.0  

dx = (Lend - Lstart) / (N-1)