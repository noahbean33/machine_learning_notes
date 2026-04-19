"""
Physics-Informed Neural Network (PINN) Solution for 2D Heat Equation.

This module implements a PINN to solve the 2D heat diffusion equation:

    ∂u/∂t = α²(∂²u/∂x² + ∂²u/∂y²)

where:
    u: temperature distribution
    t: time
    x, y: spatial coordinates
    α: thermal diffusivity coefficient

The PINN approach enforces:
- Initial condition: u(x, y, 0) = sin(πx)·sin(πy)
- Boundary conditions: u = 0 at all boundaries
"""

import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt


class HeatPINN(nn.Module):
    """Neural network for solving 2D heat equation."""
    
    def __init__(self, layers=None):
        """
        Initialize the neural network.
        
        Args:
            layers: List of layer sizes. Default: [3, 64, 64, 1]
                   Input: (x, y, t), Output: u
        """
        super(HeatPINN, self).__init__()
        
        if layers is None:
            layers = [3, 64, 64, 1]
        
        # Build sequential network
        network_layers = []
        for i in range(len(layers) - 1):
            network_layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation on output layer
                network_layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*network_layers)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3) containing [x, y, t]
        
        Returns:
            Predicted temperature u
        """
        return self.net(x)


def initial_condition(x, y):
    """
    Define initial temperature distribution.
    
    Args:
        x: X coordinates
        y: Y coordinates
    
    Returns:
        Initial temperature: sin(πx)·sin(πy)
    """
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def boundary_condition(x, y, t, value=0.0):
    """
    Define boundary temperature.
    
    Args:
        x: X coordinates
        y: Y coordinates
        t: Time coordinates
        value: Boundary temperature value
    
    Returns:
        Tensor filled with boundary value
    """
    return torch.full_like(x, value)


def generate_training_data(num_points):
    """
    Generate random training points in the domain [0,1] × [0,1] × [0,1].
    
    Args:
        num_points: Number of points to generate
    
    Returns:
        tuple: (x, y, t) coordinate tensors with gradients enabled
    """
    x = torch.rand(num_points, 1, requires_grad=True)
    y = torch.rand(num_points, 1, requires_grad=True)
    t = torch.rand(num_points, 1, requires_grad=True)
    
    return x, y, t


def generate_boundary_points(num_points):
    """
    Generate points on the spatial boundaries.
    
    Args:
        num_points: Number of boundary points
    
    Returns:
        tuple: (x_boundary, y_boundary) coordinates
    """
    # Alternate between x and y boundaries
    x_boundary = torch.tensor([0.0, 1.0]).repeat(num_points // 2)
    y_boundary = torch.rand(num_points)
    
    # Randomly switch x and y to cover all boundaries
    if torch.rand(1) > 0.5:
        x_boundary, y_boundary = y_boundary, x_boundary
    
    return x_boundary.view(-1, 1), y_boundary.view(-1, 1)


def generate_boundary_training_data(num_points):
    """
    Generate boundary points with random time values.
    
    Args:
        num_points: Number of points
    
    Returns:
        tuple: (x_boundary, y_boundary, t)
    """
    x_boundary, y_boundary = generate_boundary_points(num_points)
    t = torch.rand(num_points, 1, requires_grad=True)
    
    return x_boundary, y_boundary, t


def compute_pde_residual(x, y, t, model, alpha=1.0):
    """
    Compute the heat equation residual using automatic differentiation.
    
    The residual is: ∂u/∂t - α²(∂²u/∂x² + ∂²u/∂y²)
    
    Args:
        x, y, t: Coordinate tensors
        model: Neural network model
        alpha: Thermal diffusivity coefficient
    
    Returns:
        PDE residual tensor
    """
    # Concatenate inputs
    input_data = torch.cat([x, y, t], dim=1)
    u = model(input_data)
    
    # First-order derivatives
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_y = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]
    
    # Second-order derivatives
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0]
    
    u_yy = torch.autograd.grad(
        u_y, y, grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0]
    
    # Heat equation residual
    residual = u_t - alpha**2 * (u_xx + u_yy)
    
    return residual


def train_pinn(model, num_iterations, num_points, lr=1e-3, alpha=1.0):
    """
    Train the PINN to solve the 2D heat equation.
    
    Args:
        model: Neural network model
        num_iterations: Number of training iterations
        num_points: Number of collocation points per iteration
        lr: Learning rate
        alpha: Thermal diffusivity coefficient
    
    Returns:
        Trained model
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print("=" * 70)
    print("Training Physics-Informed Neural Network for 2D Heat Equation")
    print("=" * 70)
    print(f"Iterations: {num_iterations}")
    print(f"Collocation points: {num_points}")
    print(f"Learning rate: {lr}")
    print(f"Thermal diffusivity (α): {alpha}")
    print("=" * 70)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Generate training data
        x, y, t = generate_training_data(num_points)
        x_b, y_b, t_b = generate_boundary_training_data(num_points)
        
        # Initial condition (t=0)
        t_initial = torch.zeros_like(t)
        u_initial = initial_condition(x, y)
        
        # Boundary conditions
        u_boundary = boundary_condition(x_b, y_b, t_b, value=0.0)
        
        # Compute PDE residual
        residual = compute_pde_residual(x, y, t, model, alpha)
        
        # Compute losses
        loss_ic = criterion(
            u_initial, 
            model(torch.cat([x, y, t_initial], dim=1))
        )
        
        loss_bc = criterion(
            u_boundary, 
            model(torch.cat([x_b, y_b, t_b], dim=1))
        )
        
        loss_pde = criterion(residual, torch.zeros_like(residual))
        
        # Total loss
        total_loss = loss_ic + loss_bc + loss_pde
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration:5d} | Total Loss: {total_loss.item():.6e} | "
                  f"IC Loss: {loss_ic.item():.6e} | BC Loss: {loss_bc.item():.6e} | "
                  f"PDE Loss: {loss_pde.item():.6e}")
    
    print("\nTraining complete!")
    return model


def plot_solution(model, t_value=0.0, resolution=100, title=None):
    """
    Plot the 2D heat equation solution at a specific time.
    
    Args:
        model: Trained neural network
        t_value: Time value to plot (0 to 1)
        resolution: Grid resolution
        title: Plot title
    """
    model.eval()
    
    with torch.no_grad():
        x_vals = torch.linspace(0, 1, resolution)
        y_vals = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        t_val = torch.ones_like(X) * t_value
        
        input_data = torch.stack([
            X.flatten(), 
            Y.flatten(), 
            t_val.flatten()
        ], dim=1)
        
        solution = model(input_data).reshape(resolution, resolution)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(solution.numpy(), cmap='jet', cbar_kws={'label': 'Temperature'})
    
    if title is None:
        title = f"2D Heat Equation Solution at t = {t_value:.2f}"
    
    plt.title(title, fontsize=14)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the 2D heat equation PINN."""
    # Initialize model
    model = HeatPINN(layers=[3, 64, 64, 1])
    
    # Training parameters
    num_iterations = 10000
    num_points = 1000
    learning_rate = 1e-3
    alpha = 1.0
    
    # Train the model
    model = train_pinn(
        model, 
        num_iterations=num_iterations,
        num_points=num_points,
        lr=learning_rate,
        alpha=alpha
    )
    
    # Plot results at different times
    print("\nGenerating plots...")
    print("-" * 70)
    
    plot_solution(model, t_value=0.0, title="2D Heat Equation at t=0 (Initial Condition)")
    plot_solution(model, t_value=0.5, title="2D Heat Equation at t=0.5")
    plot_solution(model, t_value=1.0, title="2D Heat Equation at t=1.0 (Final)")


if __name__ == "__main__":
    main()
