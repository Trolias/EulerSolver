## 2D Euler Equations Solver 

This repository contains a **finite-volume 2D solver** for the compressible **Euler equations** in Cartesian coordinates:

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} + \frac{\partial \mathbf{G}(\mathbf{U})}{\partial y} = 0,
$$

where the state vector $\mathbf{U}$ and the directional flux vectors $\mathbf{F}$ and $\mathbf{G}$ are:

$$
\mathbf{U} = 
\begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
E
\end{bmatrix},
\qquad
\mathbf{F}(\mathbf{U}) = 
\begin{bmatrix}
\rho u \\
\rho u^2 + p \\
\rho uv \\
u(E+p)
\end{bmatrix},
\qquad
\mathbf{G}(\mathbf{U}) = 
\begin{bmatrix}
\rho v \\
\rho uv \\
\rho v^2 + p \\
v(E+p)
\end{bmatrix}.
$$

Equation of state:

$$
p = (\gamma - 1)\left(E - \frac{1}{2}\rho (u^2 + v^2)\right).
$$

### Features (current)
- **Time integration**
  - Forward Euler
  - TVD Runge–Kutta, 3rd order 

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
  - HLLC (Extended for transverse waves)

![Isentropic Vortex animation](\images\vor_20s.gif)

![2D Riemann problem animation](\images\Riemann_20s.gif)
