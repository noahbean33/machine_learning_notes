"""
DeepXDE Solution for 1D Heat Equation.

This module uses the DeepXDE library to solve the 1D heat diffusion equation:

    ∂u/∂t = k·∂²u/∂x²

where:
    u: temperature distribution
    t: time
    x: spatial coordinate
    k: thermal diffusivity

DeepXDE provides high-level APIs for:
- Automated PINN architecture
- Built-in geometry handling
- Boundary and initial condition specification
- Training data logging
"""

import deepxde as dde
from deepxde.backend import tf
import numpy as np
import matplotlib.pyplot as plt


def pde_heat_1d(comp_space, u, k=0.4):
    """
    Define the 1D heat equation PDE.
    
    Args:
        comp_space: Computational space tensor [x, t]
        u: Solution tensor
        k: Thermal diffusivity coefficient
    
    Returns:
        PDE residual: ∂u/∂t - k·∂²u/∂x²
    """
    # First-order derivative w.r.t. time (j=1)
    du_t = dde.grad.jacobian(u, comp_space, i=0, j=1)
    
    # Second-order derivative w.r.t. space (i=0, j=0)
    du_xx = dde.grad.hessian(u, comp_space, i=0, j=0)
    
    # Return residual
    return du_t - k * du_xx


def create_geometry_and_conditions(L=1.0, n=1.0, k=0.4):
    """
    Create geometry and boundary/initial conditions for the heat equation.
    
    Args:
        L: Length of spatial domain
        n: Length of temporal domain
        k: Thermal diffusivity
    
    Returns:
        tuple: (geomtime, bc, ic) - geometry and conditions
    """
    # Define spatial and temporal domains
    geom = dde.geometry.Interval(0, L)
    timedomain = dde.geometry.TimeDomain(0, n)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    # Define initial condition: u(x, 0) = sin(πx/L)
    ic = dde.icbc.IC(
        geomtime,
        lambda x: np.sin(n * np.pi * x[:, 0:1] / L),
        lambda _, on_initial: on_initial
    )
    
    # Define boundary condition: u = 0 at boundaries
    bc = dde.icbc.DirichletBC(
        geomtime,
        lambda x: 0.0,
        lambda _, on_boundary: on_boundary
    )
    
    return geomtime, bc, ic


def create_pde_data(geomtime, pde_func, bc, ic, num_domain=2540, 
                    num_boundary=80, num_initial=160, num_test=2540):
    """
    Create PDE training data.
    
    Args:
        geomtime: Geometry-time domain
        pde_func: PDE function
        bc: Boundary condition
        ic: Initial condition
        num_domain: Number of domain collocation points
        num_boundary: Number of boundary points
        num_initial: Number of initial condition points
        num_test: Number of test points
    
    Returns:
        DeepXDE TimePDE data object
    """
    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        [bc, ic],
        num_domain=num_domain,
        num_boundary=num_boundary,
        num_initial=num_initial,
        num_test=num_test
    )
    
    return data


def visualize_training_points(data):
    """
    Visualize the distribution of training points in space-time.
    
    Args:
        data: DeepXDE data object
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(
        data.train_x_all[:, 0],  # x coordinates
        data.train_x_all[:, 1],  # t coordinates
        s=1,
        alpha=0.5
    )
    plt.xlabel('Spatial Coordinate (x)', fontsize=12)
    plt.ylabel('Time Coordinate (t)', fontsize=12)
    plt.title('Distribution of Training Points (Meshless PINN)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_model(data, net, adam_iterations=15000, lbfgs=True):
    """
    Train the PINN model using Adam and optionally L-BFGS.
    
    Args:
        data: Training data
        net: Neural network architecture
        adam_iterations: Number of Adam optimizer iterations
        lbfgs: Whether to use L-BFGS for refinement
    
    Returns:
        tuple: (model, losshistory, train_state)
    """
    # Initialize model
    model = dde.Model(data, net)
    
    # Phase 1: Adam optimization
    print("=" * 70)
    print("Training with Adam optimizer")
    print("=" * 70)
    model.compile("adam", lr=1e-3)
    losshistory, train_state = model.train(iterations=adam_iterations)
    
    # Phase 2: L-BFGS optimization (for better accuracy)
    if lbfgs:
        print("\n" + "=" * 70)
        print("Refining with L-BFGS optimizer")
        print("=" * 70)
        model.compile("L-BFGS")
        losshistory, train_state = model.train()
    
    print("\nTraining complete!")
    return model, losshistory, train_state


def main():
    """Main function to solve 1D heat equation using DeepXDE."""
    # Problem parameters
    k = 0.4   # Thermal diffusivity
    L = 1.0   # Spatial domain length
    n = 1.0   # Temporal domain length
    
    print("=" * 70)
    print("DeepXDE Solution for 1D Heat Equation")
    print("=" * 70)
    print(f"Thermal diffusivity (k): {k}")
    print(f"Spatial domain: [0, {L}]")
    print(f"Temporal domain: [0, {n}]")
    print("=" * 70)
    
    # Create PDE function with fixed k
    pde_func = lambda comp_space, u: pde_heat_1d(comp_space, u, k=k)
    
    # Create geometry and conditions
    print("\nCreating geometry and conditions...")
    geomtime, bc, ic = create_geometry_and_conditions(L, n, k)
    
    # Create training data
    print("Creating training data...")
    data = create_pde_data(
        geomtime, pde_func, bc, ic,
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
        num_test=2540
    )
    
    # Visualize training points
    print("\nVisualizing training point distribution...")
    visualize_training_points(data)
    
    # Define neural network architecture
    # Input: 2 nodes (x, t)
    # Hidden: 3 layers with 20 nodes each
    # Output: 1 node (u)
    print("\nCreating neural network...")
    net = dde.nn.FNN(
        [2] + 3 * [20] + [1],
        "tanh",
        "Glorot normal"
    )
    
    # Train the model
    print("\nStarting training...\n")
    model, losshistory, train_state = train_model(
        data, net,
        adam_iterations=15000,
        lbfgs=True
    )
    
    # Save results
    print("\nSaving results...")
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    print("\n" + "=" * 70)
    print("Simulation complete! Results saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
