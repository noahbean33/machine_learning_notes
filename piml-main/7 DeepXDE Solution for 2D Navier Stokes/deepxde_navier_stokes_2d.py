"""
DeepXDE Solution for 2D Navier-Stokes Equations.

This module uses DeepXDE to solve the steady-state 2D Navier-Stokes equations:

Momentum equation for U:
    U·∂U/∂x + V·∂U/∂y = -(1/ρ)·∂P/∂x + ν·(∂²U/∂x² + ∂²U/∂y²)

Momentum equation for V:
    U·∂V/∂x + V·∂V/∂y = -(1/ρ)·∂P/∂y + ν·(∂²V/∂x² + ∂²V/∂y²)

Continuity equation:
    ∂U/∂x + ∂V/∂y = 0

where:
    U, V: velocity components in x and y directions
    P: pressure
    ρ: fluid density
    ν: kinematic viscosity (ν = μ/ρ)
"""

import tensorflow as tf
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt


def create_geometry(L=2.0, D=1.0):
    """
    Create rectangular geometry for the flow domain.
    
    Args:
        L: Length of domain (x-direction)
        D: Width of domain (y-direction)
    
    Returns:
        DeepXDE Rectangle geometry centered at origin
    """
    geom = dde.geometry.Rectangle(
        xmin=[-L/2, -D/2],  # Lower-left corner
        xmax=[L/2, D/2]      # Upper-right corner
    )
    return geom


def boundary_wall(X, on_boundary, D=1.0):
    """
    Check if point is on wall boundaries (top or bottom).
    
    Args:
        X: Point coordinates [x, y]
        on_boundary: Boolean indicating if point is on any boundary
        D: Domain width
    
    Returns:
        Boolean indicating if point is on wall
    """
    on_wall = np.logical_and(
        np.logical_or(
            np.isclose(X[1], -D/2, rtol=1e-5, atol=1e-8),  # Bottom wall
            np.isclose(X[1], D/2, rtol=1e-5, atol=1e-8)    # Top wall
        ),
        on_boundary
    )
    return on_wall


def boundary_inlet(X, on_boundary, L=2.0):
    """
    Check if point is on inlet boundary (left side).
    
    Args:
        X: Point coordinates [x, y]
        on_boundary: Boolean indicating if point is on any boundary
        L: Domain length
    
    Returns:
        Boolean indicating if point is on inlet
    """
    on_inlet = np.logical_and(
        np.isclose(X[0], -L/2, rtol=1e-5, atol=1e-8),
        on_boundary
    )
    return on_inlet


def boundary_outlet(X, on_boundary, L=2.0):
    """
    Check if point is on outlet boundary (right side).
    
    Args:
        X: Point coordinates [x, y]
        on_boundary: Boolean indicating if point is on any boundary
        L: Domain length
    
    Returns:
        Boolean indicating if point is on outlet
    """
    on_outlet = np.logical_and(
        np.isclose(X[0], L/2, rtol=1e-5, atol=1e-8),
        on_boundary
    )
    return on_outlet


def create_boundary_conditions(geom, u_in=1.0, L=2.0, D=1.0):
    """
    Create boundary conditions for Navier-Stokes equations.
    
    Args:
        geom: Geometry object
        u_in: Inlet velocity
        L: Domain length
        D: Domain width
    
    Returns:
        List of boundary condition objects
    """
    # Wall boundaries (no-slip condition: u = v = 0)
    bc_wall_u = dde.DirichletBC(
        geom, lambda X: 0.0,
        lambda X, on_boundary: boundary_wall(X, on_boundary, D),
        component=0  # U velocity
    )
    bc_wall_v = dde.DirichletBC(
        geom, lambda X: 0.0,
        lambda X, on_boundary: boundary_wall(X, on_boundary, D),
        component=1  # V velocity
    )
    
    # Inlet boundary (prescribed velocity)
    bc_inlet_u = dde.DirichletBC(
        geom, lambda X: u_in,
        lambda X, on_boundary: boundary_inlet(X, on_boundary, L),
        component=0  # U velocity
    )
    bc_inlet_v = dde.DirichletBC(
        geom, lambda X: 0.0,
        lambda X, on_boundary: boundary_inlet(X, on_boundary, L),
        component=1  # V velocity
    )
    
    # Outlet boundary (zero pressure and v velocity)
    bc_outlet_p = dde.DirichletBC(
        geom, lambda X: 0.0,
        lambda X, on_boundary: boundary_outlet(X, on_boundary, L),
        component=2  # Pressure
    )
    bc_outlet_v = dde.DirichletBC(
        geom, lambda X: 0.0,
        lambda X, on_boundary: boundary_outlet(X, on_boundary, L),
        component=1  # V velocity
    )
    
    return [bc_wall_u, bc_wall_v, bc_inlet_u, bc_inlet_v, bc_outlet_p, bc_outlet_v]


def pde_navier_stokes(X, Y, rho=1.0, mu=1.0):
    """
    Define the Navier-Stokes PDE system.
    
    Args:
        X: Input coordinates [x, y]
        Y: Output variables [U, V, P]
        rho: Fluid density
        mu: Dynamic viscosity
    
    Returns:
        List of PDE residuals [momentum_u, momentum_v, continuity]
    """
    # Extract velocity and pressure components
    # Y[:, 0:1] = U, Y[:, 1:2] = V, Y[:, 2:3] = P
    
    # First-order derivatives
    du_x = dde.grad.jacobian(Y, X, i=0, j=0)  # ∂U/∂x
    du_y = dde.grad.jacobian(Y, X, i=0, j=1)  # ∂U/∂y
    dv_x = dde.grad.jacobian(Y, X, i=1, j=0)  # ∂V/∂x
    dv_y = dde.grad.jacobian(Y, X, i=1, j=1)  # ∂V/∂y
    dp_x = dde.grad.jacobian(Y, X, i=2, j=0)  # ∂P/∂x
    dp_y = dde.grad.jacobian(Y, X, i=2, j=1)  # ∂P/∂y
    
    # Second-order derivatives
    du_xx = dde.grad.hessian(Y, X, component=0, i=0, j=0)  # ∂²U/∂x²
    du_yy = dde.grad.hessian(Y, X, component=0, i=1, j=1)  # ∂²U/∂y²
    dv_xx = dde.grad.hessian(Y, X, component=1, i=0, j=0)  # ∂²V/∂x²
    dv_yy = dde.grad.hessian(Y, X, component=1, i=1, j=1)  # ∂²V/∂y²
    
    # Kinematic viscosity
    nu = mu / rho
    
    # Momentum equation for U (steady-state, so ∂U/∂t = 0)
    pde_u = (
        Y[:, 0:1] * du_x + Y[:, 1:2] * du_y +  # Convection terms
        (1/rho) * dp_x -                        # Pressure gradient
        nu * (du_xx + du_yy)                    # Diffusion terms
    )
    
    # Momentum equation for V
    pde_v = (
        Y[:, 0:1] * dv_x + Y[:, 1:2] * dv_y +  # Convection terms
        (1/rho) * dp_y -                        # Pressure gradient
        nu * (dv_xx + dv_yy)                    # Diffusion terms
    )
    
    # Continuity equation (mass conservation)
    pde_cont = du_x + dv_y
    
    return [pde_u, pde_v, pde_cont]


def visualize_training_points(data, L=2.0, D=1.0):
    """
    Visualize the distribution of training points.
    
    Args:
        data: DeepXDE PDE data object
        L: Domain length
        D: Domain width
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        data.train_x_all[:, 0],
        data.train_x_all[:, 1],
        s=1,
        alpha=0.5
    )
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Distribution of Training Points (Meshless)', fontsize=14)
    plt.xlim((-L/2, L/2))
    plt.ylim((-D/2, D/2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_results(model, geom, L=2.0, D=1.0, num_samples=500000):
    """
    Generate and plot the solution fields.
    
    Args:
        model: Trained DeepXDE model
        geom: Geometry object
        L: Domain length
        D: Domain width
        num_samples: Number of points for visualization
    """
    # Generate test points
    samples = geom.random_points(num_samples)
    result = model.predict(samples)
    
    # Color scale limits for each variable
    color_limits = [
        [0, 1.5],      # U velocity
        [-0.3, 0.3],   # V velocity
        [0, 35]        # Pressure (scaled)
    ]
    
    labels = ['U Velocity', 'V Velocity', 'Pressure']
    
    # Plot each field
    for idx in range(3):
        plt.figure(figsize=(12, 4))
        scatter = plt.scatter(
            samples[:, 0],
            samples[:, 1],
            c=result[:, idx],
            cmap='jet',
            s=2,
            vmin=color_limits[idx][0],
            vmax=color_limits[idx][1]
        )
        plt.colorbar(scatter, label=labels[idx])
        plt.xlim((-L/2, L/2))
        plt.ylim((-D/2, D/2))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.title(f'2D Navier-Stokes Solution: {labels[idx]}', fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to solve 2D Navier-Stokes equations using DeepXDE."""
    # Physical parameters
    rho = 1.0    # Fluid density
    mu = 1.0     # Dynamic viscosity
    u_in = 1.0   # Inlet velocity
    
    # Domain parameters
    L = 2.0      # Length
    D = 1.0      # Width
    
    print("=" * 70)
    print("DeepXDE Solution for 2D Navier-Stokes Equations")
    print("=" * 70)
    print(f"Fluid density (ρ): {rho}")
    print(f"Dynamic viscosity (μ): {mu}")
    print(f"Kinematic viscosity (ν): {mu/rho}")
    print(f"Inlet velocity: {u_in}")
    print(f"Domain: [{-L/2}, {L/2}] × [{-D/2}, {D/2}]")
    print("=" * 70)
    
    # Create geometry
    print("\nCreating geometry...")
    geom = create_geometry(L, D)
    
    # Create boundary conditions
    print("Creating boundary conditions...")
    boundary_conditions = create_boundary_conditions(geom, u_in, L, D)
    
    # Create PDE function
    pde_func = lambda X, Y: pde_navier_stokes(X, Y, rho, mu)
    
    # Create PDE data
    print("Creating PDE data...")
    data = dde.data.PDE(
        geom,
        pde_func,
        boundary_conditions,
        num_domain=2000,
        num_boundary=200,
        num_test=200
    )
    
    # Visualize training points
    print("\nVisualizing training point distribution...")
    visualize_training_points(data, L, D)
    
    # Define neural network
    # Input: 2 nodes (x, y)
    # Hidden: 5 layers with 64 nodes each
    # Output: 3 nodes (U, V, P)
    print("\nCreating neural network...")
    net = dde.maps.FNN(
        [2] + 5 * [64] + [3],
        "tanh",
        "Glorot uniform"
    )
    
    # Initialize and compile model
    print("\nInitializing model...")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    
    # Train with Adam
    print("\n" + "=" * 70)
    print("Training with Adam optimizer (may take ~15 minutes)")
    print("=" * 70)
    losshistory, train_state = model.train(epochs=10000)
    
    # Refine with L-BFGS
    print("\n" + "=" * 70)
    print("Refining with L-BFGS optimizer (may take ~5 minutes)")
    print("=" * 70)
    dde.optimizers.config.set_LBFGS_options(maxiter=3000)
    model.compile("L-BFGS")
    losshistory, train_state = model.train()
    
    # Save training history
    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    
    print("\nTraining complete!")
    
    # Generate and plot results
    print("\n" + "=" * 70)
    print("Generating solution visualization...")
    print("=" * 70)
    plot_results(model, geom, L, D, num_samples=500000)
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
