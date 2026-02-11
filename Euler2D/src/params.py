Nx = 801
Ny = 801
gamma = 1.4
CFL = 0.4

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

shm = 3
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

cas = 1
case = {
    1: "2D Riemann Problem",
    2: "2D Sod Shock",
    3: "Explosion"
}

if cas == 1:   
    Lxstart = -0.5
    Lxend = 0.5
    Lystart = -0.5
    Lyend = 0.5
    t_end = 0.4
elif cas == 2:  
    Lxstart = 0.0  
    Lxend = 1.0
    Lystart = 0.0
    Lyend = 0.2
    t_end = 0.2
elif cas == 3:
    Lxstart = -0.75
    Lxend = 0.75
    Lystart = -0.75
    Lyend = 0.75
    t_end = 0.25


dx = (Lxend - Lxstart) / (Nx-1)
dy = (Lyend - Lystart) / (Ny-1)