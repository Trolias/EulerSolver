import numpy as np
import matplotlib.pyplot as plt

def save_dat(filename, x, PV):

    x_np = np.asarray(x)
    PV_np = np.asarray(PV)
    data = np.column_stack([x_np, PV_np[:,0], PV_np[:,1], PV_np[:,2]])
    header = "#x   rho   u   p"
    np.savetxt(filename, data, header=header, comments='')
    print(f"Saved: {filename}  (shape={data.shape})")

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
