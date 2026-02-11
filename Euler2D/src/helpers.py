import numpy as np
import matplotlib.pyplot as plt

import numpy as np

def save_tecplot_dat(filename, x, y, PV, title="2D Euler", zone_name="Zone 1"):
    """
    Write Tecplot ASCII .dat for a structured 2D grid readable in ParaView.
    Variables: x, y, rho, u, v, p

    Accepts:
      x: (Nx,) or (Ny,Nx) or (Nx,Ny)
      y: (Ny,) or (Ny,Nx) or (Nx,Ny)
      PV: (Ny,Nx,4) or (Nx,Ny,4) with PV[...,0]=rho,1=u,2=v,3=p

    Output: Tecplot ASCII, DATAPACKING=POINT, ZONETYPE=ORDERED
    """

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    PV_np = np.asarray(PV)

    # --- Build X,Y as 2D arrays (Ny, Nx) ---
    if x_np.ndim == 1 and y_np.ndim == 1:
        # common case: x=(Nx,), y=(Ny,)
        Nx = x_np.size
        Ny = y_np.size
        X, Y = np.meshgrid(x_np, y_np)  # default indexing='xy' -> X,Y shapes (Ny,Nx)
    else:
        # assume already on a grid
        X = x_np
        Y = y_np
        if X.shape != Y.shape:
            raise ValueError(f"x and y grids must have same shape, got {X.shape} vs {Y.shape}")
        Ny, Nx = X.shape  # expecting (Ny,Nx)

    # --- Ensure PV is (Ny, Nx, 4) ---
    if PV_np.ndim != 3 or PV_np.shape[-1] != 4:
        raise ValueError(f"PV must be (Ny,Nx,4); got {PV_np.shape}")

    if PV_np.shape[0] == Nx and PV_np.shape[1] == Ny:
        # PV is (Nx,Ny,4) -> transpose to (Ny,Nx,4)
        PV_np = np.transpose(PV_np, (1, 0, 2))

    if PV_np.shape[0] != Ny or PV_np.shape[1] != Nx:
        raise ValueError(f"PV shape {PV_np.shape} not compatible with grid {(Ny,Nx)}")

    # Flatten in row-major order (y first, then x), consistent with meshgrid() above
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    PVf = PV_np.reshape(-1, 4)

    # --- Write Tecplot ASCII ---
    with open(filename, "w") as f:
        f.write(f'TITLE = "{title}"\n')
        f.write('VARIABLES = "x" "y" "rho" "u" "v" "p"\n')
        f.write(f'ZONE T="{zone_name}", I={Nx}, J={Ny}, DATAPACKING=POINT\n')
        # One point per line: x y rho u v p
        for i in range(Xf.size):
            f.write(f"{Xf[i]:.16e} {Yf[i]:.16e} {PVf[i,0]:.16e} {PVf[i,1]:.16e} {PVf[i,2]:.16e} {PVf[i,3]:.16e}\n")

    print(f"Saved Tecplot ASCII: {filename} (I={Nx}, J={Ny}, N={Nx*Ny})")


def load_dat(filename):

    data = np.loadtxt(filename, comments='#')
    x   = data[:,0]
    rho = data[:,1]
    u   = data[:,2]
    p   = data[:,3]
    return x, rho, u, p

def plot_compare(dat_a, dat_b, label_a="A", label_b="B"):
    xa, rho_a, u_a, p_a = load_dat(dat_a)
    xb, rho_b, u_b, p_b = load_dat(dat_b)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(xa, rho_a, label=label_a)
    axs[0].plot(xb, rho_b, '--', label=label_b)
    axs[0].set_ylabel("rho")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(xa, u_a, label=label_a)
    axs[1].plot(xb, u_b, '--', label=label_b)
    axs[1].set_ylabel("u")
    axs[1].grid(True)

    axs[2].plot(xa, p_a, label=label_a)
    axs[2].plot(xb, p_b, '--', label=label_b)
    axs[2].set_ylabel("p")
    axs[2].set_xlabel("x")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
