"""
Finite Difference Method (FDM) Solution for 1D Heat Equation.

This module implements a numerical solution for the 1D heat diffusion equation
using an explicit finite difference scheme. The heat equation is given by:

    ∂u/∂t = k ∂²u/∂x²

where:
    u: temperature distribution
    t: time
    x: spatial coordinate
    k: thermal diffusivity (heat capacity)
"""

import numpy as np
from matplotlib import pyplot as plt


def initialize_grid(length, dx, total_time, dt, temp_left, temp_right):
    """
    Initialize the spatial and temporal grids with boundary conditions.
    
    Args:
        length: Length of the rod in meters
        dx: Spatial step size
        total_time: Total simulation time
        dt: Time step size
        temp_left: Temperature at the left boundary
        temp_right: Temperature at the right boundary
    
    Returns:
        tuple: (x_vector, t_vector, u) where:
            - x_vector: Spatial grid points
            - t_vector: Temporal grid points
            - u: Temperature matrix initialized with boundary conditions
    """
    # Create spatial grid
    x_vector = np.linspace(0, length, int(length / dx))
    
    # Create temporal grid
    t_vector = np.linspace(0, total_time, int(total_time / dt))
    
    # Initialize temperature matrix: rows = time steps, columns = spatial points
    u = np.zeros([len(t_vector), len(x_vector)])
    
    # Apply boundary conditions
    u[:, 0] = temp_left   # Left boundary
    u[:, -1] = temp_right  # Right boundary
    
    return x_vector, t_vector, u


def solve_heat_equation_1d(u, x_vector, t_vector, k, dx, dt):
    """
    Solve the 1D heat equation using explicit finite difference method.
    
    The discretized equation is:
        u(t+dt, x) = u(t,x) + k * (u(t,x+dx) - 2*u(t,x) + u(t,x-dx)) / dx² * dt
    
    Args:
        u: Temperature matrix (initialized with boundary conditions)
        x_vector: Spatial grid points
        t_vector: Temporal grid points
        k: Thermal diffusivity
        dx: Spatial step size
        dt: Time step size
    
    Returns:
        np.ndarray: Updated temperature matrix
    """
    # Precompute constant factor for performance
    const = k * (dt / dx**2)
    
    # Iterate through time steps
    # Start from 1 to avoid boundary condition issues
    for t in range(1, len(t_vector) - 1):
        # Iterate through spatial points (excluding boundaries)
        for x in range(1, len(x_vector) - 1):
            # Apply finite difference scheme
            u[t + 1, x] = const * (u[t, x + 1] - 2 * u[t, x] + u[t, x - 1]) + u[t, x]
    
    return u


def plot_temperature_distribution(x_vector, u, time_steps, total_time, dt):
    """
    Plot temperature distribution at different time steps.
    
    Args:
        x_vector: Spatial grid points
        u: Temperature matrix
        time_steps: List of time step indices to plot
        total_time: Total simulation time
        dt: Time step size
    """
    plt.figure(figsize=(10, 6))
    
    for t_idx in time_steps:
        # Calculate actual time from index
        actual_time = t_idx * dt
        plt.plot(x_vector, u[t_idx], label=f't = {actual_time:.2f}s')
    
    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('1D Heat Equation Solution - Temperature Distribution Over Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the 1D heat equation simulation."""
    # Physical parameters
    length = 10.0  # Length of rod in meters
    k = 0.89       # Thermal diffusivity (aluminum)
    
    # Boundary conditions
    temp_left = 100.0   # Temperature at left end (°C)
    temp_right = 200.0  # Temperature at right end (°C)
    
    # Simulation parameters
    total_sim_time = 10.0  # Total simulation time in seconds
    dx = 0.1               # Spatial step size
    dt = 0.0001            # Time step size
    
    print("=" * 60)
    print("1D Heat Equation - Finite Difference Method")
    print("=" * 60)
    print(f"Rod length: {length} m")
    print(f"Thermal diffusivity (k): {k} J/(g·°C)")
    print(f"Boundary conditions: T_left = {temp_left}°C, T_right = {temp_right}°C")
    print(f"Spatial step size (dx): {dx} m")
    print(f"Time step size (dt): {dt} s")
    print(f"Total time steps: {int(total_sim_time / dt)}")
    print("=" * 60)
    
    # Initialize grids
    x_vector, t_vector, u = initialize_grid(
        length, dx, total_sim_time, dt, temp_left, temp_right
    )
    
    # Solve the heat equation
    print("\nSolving heat equation...")
    u = solve_heat_equation_1d(u, x_vector, t_vector, k, dx, dt)
    print("Solution complete!")
    
    # Plot results at selected time steps
    time_step_indices = [0, 500, 5000, 50000, 99999]
    print(f"\nPlotting temperature distribution at time steps: {time_step_indices}")
    plot_temperature_distribution(x_vector, u, time_step_indices, total_sim_time, dt)


if __name__ == "__main__":
    main()
