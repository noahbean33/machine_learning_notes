"""
Finite Difference Method (FDM) Solution for 2D Burgers Equation.

This module implements a numerical solution for the viscous Burgers equation
in 2D using the finite difference method. The Burgers equations are:

Horizontal velocity (u):
    ∂u/∂t + u·∂u/∂x + v·∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)

Vertical velocity (v):
    ∂v/∂t + u·∂v/∂x + v·∂v/∂y = ν(∂²v/∂x² + ∂²v/∂y²)

where:
    u, v: velocity components in x and y directions
    ν: kinematic viscosity (diffusion coefficient)
    t: time
"""

import numpy as np
import matplotlib.pyplot as plt


def initialize_velocity_fields(nx, ny, nt, dx, dy):
    """
    Initialize velocity fields with initial conditions.
    
    Args:
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction
        nt: Number of time steps
        dx: Spatial step size in x direction
        dy: Spatial step size in y direction
    
    Returns:
        tuple: (u, v, uf, vf) initialized velocity arrays
    """
    # Initialize velocity fields
    u = np.ones((ny, nx))
    v = np.ones((ny, nx))
    uf = np.ones((nt, ny, nx))
    vf = np.ones((nt, ny, nx))
    
    # Set initial high-speed region [0.75, 1.25] x [0.75, 1.25]
    y_start = int(0.75 / dy)
    y_end = int(1.25 / dy + 1)
    x_start = int(0.75 / dx)
    x_end = int(1.25 / dx + 1)
    
    u[y_start:y_end, x_start:x_end] = 5.0
    v[y_start:y_end, x_start:x_end] = 5.0
    uf[0, y_start:y_end, x_start:x_end] = 5.0
    vf[0, y_start:y_end, x_start:x_end] = 5.0
    
    return u, v, uf, vf


def apply_boundary_conditions(u, v):
    """
    Apply Dirichlet boundary conditions (u = v = 1 at all boundaries).
    
    Args:
        u: Horizontal velocity field
        v: Vertical velocity field
    
    Returns:
        tuple: Updated (u, v) with boundary conditions applied
    """
    u[:, 0] = 1.0   # Left boundary
    u[:, -1] = 1.0  # Right boundary
    u[0, :] = 1.0   # Bottom boundary
    u[-1, :] = 1.0  # Top boundary
    
    v[:, 0] = 1.0
    v[:, -1] = 1.0
    v[0, :] = 1.0
    v[-1, :] = 1.0
    
    return u, v


def solve_burgers_2d(u, v, uf, vf, nx, ny, nt, dx, dy, dt, nu):
    """
    Solve the 2D Burgers equations using explicit finite difference method.
    
    Args:
        u: Initial horizontal velocity field
        v: Initial vertical velocity field
        uf: Storage for u solution over time
        vf: Storage for v solution over time
        nx: Number of grid points in x
        ny: Number of grid points in y
        nt: Number of time steps
        dx: Spatial step size in x
        dy: Spatial step size in y
        dt: Time step size
        nu: Kinematic viscosity
    
    Returns:
        tuple: (u, v, uf, vf) solution arrays
    """
    # Precompute constant factors for performance
    dt_dx = dt / dx
    dt_dy = dt / dy
    nu_dt_dx2 = nu * dt / (dx**2)
    nu_dt_dy2 = nu * dt / (dy**2)
    
    # Temporary arrays for previous time step
    un = np.zeros((ny, nx))
    vn = np.zeros((ny, nx))
    
    print(f"Solving 2D Burgers equations for {nt} time steps...")
    
    # Time stepping loop
    for n in range(1, nt):
        # Save previous time step
        un = u.copy()
        vn = v.copy()
        
        # Spatial loop (interior points only)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                un_ij = un[i, j]
                vn_ij = vn[i, j]
                
                # Update u using finite difference scheme
                # Convection terms + diffusion terms
                convection_u = (
                    un_ij * dt_dx * (un_ij - un[i - 1, j]) +
                    vn_ij * dt_dy * (un_ij - un[i, j - 1])
                )
                diffusion_u = (
                    nu_dt_dx2 * (un[i + 1, j] - 2 * un_ij + un[i - 1, j]) +
                    nu_dt_dy2 * (un[i, j + 1] - 2 * un_ij + un[i, j - 1])
                )
                u[i, j] = un_ij - convection_u + diffusion_u
                
                # Update v using finite difference scheme
                convection_v = (
                    un_ij * dt_dx * (vn_ij - vn[i - 1, j]) +
                    vn_ij * dt_dy * (vn_ij - vn[i, j - 1])
                )
                diffusion_v = (
                    nu_dt_dx2 * (vn[i + 1, j] - 2 * vn_ij + vn[i - 1, j]) +
                    nu_dt_dy2 * (vn[i, j + 1] - 2 * vn_ij + vn[i, j - 1])
                )
                v[i, j] = vn_ij - convection_v + diffusion_v
                
                # Store solution
                uf[n, i, j] = u[i, j]
                vf[n, i, j] = v[i, j]
        
        # Apply boundary conditions
        u, v = apply_boundary_conditions(u, v)
        
        # Progress indicator
        if n % 100 == 0:
            print(f"  Time step {n}/{nt} completed")
    
    print("Solution complete!")
    return u, v, uf, vf


def plot_velocity_field(x, y, field, title, figsize=(8, 6)):
    """
    Plot a 2D velocity field using contour plot.
    
    Args:
        x: X-axis grid points
        y: Y-axis grid points
        field: 2D velocity field to plot
        title: Plot title
        figsize: Figure size tuple
    """
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=figsize)
    contour = plt.contourf(X, Y, field, levels=20, cmap='jet')
    plt.title(title, fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    colorbar = plt.colorbar(contour)
    colorbar.set_label("Velocity", fontsize=11)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the 2D Burgers equation simulation."""
    # Simulation parameters
    nt = 500        # Number of time steps
    nx = 51         # Grid points in x direction
    ny = 51         # Grid points in y direction
    nu = 0.1        # Kinematic viscosity
    dt = 0.001      # Time step size
    
    # Domain parameters
    x_domain_max = 2.0
    y_domain_max = 2.0
    dx = x_domain_max / (nx - 1)
    dy = y_domain_max / (ny - 1)
    
    print("=" * 60)
    print("2D Burgers Equation - Finite Difference Method")
    print("=" * 60)
    print(f"Domain: [{0}, {x_domain_max}] x [{0}, {y_domain_max}]")
    print(f"Grid size: {nx} x {ny}")
    print(f"Time steps: {nt}")
    print(f"Kinematic viscosity (ν): {nu}")
    print(f"Step sizes: dx = {dx:.4f}, dy = {dy:.4f}, dt = {dt}")
    print("=" * 60)
    
    # Create spatial grids
    x = np.linspace(0, x_domain_max, nx)
    y = np.linspace(0, y_domain_max, ny)
    
    # Initialize velocity fields
    u, v, uf, vf = initialize_velocity_fields(nx, ny, nt, dx, dy)
    
    # Plot initial conditions
    print("\nPlotting initial conditions...")
    plot_velocity_field(x, y, u, "Initial Horizontal Velocity (u)")
    plot_velocity_field(x, y, v, "Initial Vertical Velocity (v)")
    
    # Solve the Burgers equations
    print()
    u, v, uf, vf = solve_burgers_2d(u, v, uf, vf, nx, ny, nt, dx, dy, dt, nu)
    
    # Plot final solution
    print("\nPlotting final solution...")
    plot_velocity_field(x, y, u, "Final Horizontal Velocity (u)")
    plot_velocity_field(x, y, v, "Final Vertical Velocity (v)")
    
    # Plot solution at intermediate time step
    t_intermediate = 30
    print(f"\nPlotting solution at time step {t_intermediate}...")
    plot_velocity_field(x, y, uf[t_intermediate, :, :], 
                       f"Horizontal Velocity (u) at t={t_intermediate*dt:.3f}s")


if __name__ == "__main__":
    main()
