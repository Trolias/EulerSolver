## 1D Euler Equations Solver (Work in Progress)

This repository contains a **finite-volume 1D solver** for the compressible **Euler equations**:
$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = 0,
\quad
\mathbf{U} =
\begin{bmatrix}
\rho \\ \rho u \\ E
\end{bmatrix},
\quad
\mathbf{F}(\mathbf{U}) =
\begin{bmatrix}
\rho u \\ \rho u^2 + p \\ u(E+p)
\end{bmatrix},
\quad
p = (\gamma - 1)\left(E - \tfrac{1}{2}\rho u^2\right)$$

### Features (current)
- **Time integration**
  - Forward Euler
  - TVD Runge–Kutta, 3rd order (SSP-RK3)

- **Flux limiters**
  - Minmod

- **Face reconstruction**
  - Piecewise constant (1st order)
  - MUSCL (2nd order)
  - WENO5 (5th order)
  - WENO5 with **projection to characteristic variables**

- **Approximate Riemann solvers**
  - Rusanov (Local Lax–Friedrichs)
  - HLL
  - HLLC

