"""
Physics-Informed Neural Network (PINN) Solution for 1D Burgers Equation.

This module implements a PINN to solve the 1D viscous Burgers equation:

    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

where:
    u: velocity field
    t: time
    x: spatial coordinate
    ν: kinematic viscosity

The PINN approach combines:
- Data loss: Fitting initial and boundary conditions
- Physics loss: Satisfying the PDE using automatic differentiation
"""

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class BurgersNN(nn.Module):
    """Neural network architecture for solving Burgers equation."""
    
    def __init__(self, layers=None):
        """
        Initialize the neural network.
        
        Args:
            layers: List of layer sizes. Default: [2, 20, 30, 30, 20, 20, 1]
        """
        super(BurgersNN, self).__init__()
        
        if layers is None:
            layers = [2, 20, 30, 30, 20, 20, 1]
        
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
            x: Input tensor of shape (batch_size, 2) containing [x, t] coordinates
        
        Returns:
            Predicted solution u
        """
        return self.net(x)


class BurgersPINN:
    """Physics-Informed Neural Network for solving 1D Burgers equation."""
    
    def __init__(self, x_range=(-1, 1), t_range=(0, 1), h=0.1, k=0.1, nu=0.01):
        """
        Initialize the PINN.
        
        Args:
            x_range: Spatial domain range (x_min, x_max)
            t_range: Temporal domain range (t_min, t_max)
            h: Spatial step size
            k: Temporal step size
            nu: Kinematic viscosity
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BurgersNN().to(self.device)
        self.nu = nu
        
        # Create computational domain
        x = torch.arange(x_range[0], x_range[1] + h, h)
        t = torch.arange(t_range[0], t_range[1] + k, k)
        self.X = torch.stack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T
        
        # Define boundary and initial conditions
        bc1 = torch.stack(torch.meshgrid(x[0], t, indexing='ij')).reshape(2, -1).T   # Left boundary
        bc2 = torch.stack(torch.meshgrid(x[-1], t, indexing='ij')).reshape(2, -1).T  # Right boundary
        ic = torch.stack(torch.meshgrid(x, t[0], indexing='ij')).reshape(2, -1).T    # Initial condition
        
        self.X_train = torch.cat([bc1, bc2, ic])
        
        # Set training data values
        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))
        y_ic = -torch.sin(math.pi * ic[:, 0])  # Initial condition: u(x, 0) = -sin(πx)
        
        self.y_train = torch.cat([y_bc1, y_bc2, y_ic]).unsqueeze(1)
        
        # Move to device and enable gradients
        self.X = self.X.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X.requires_grad = True
        
        # Initialize optimizers
        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        
        self.criterion = nn.MSELoss()
        self.iter = 1
    
    def compute_pde_loss(self):
        """
        Compute the PDE residual loss using automatic differentiation.
        
        Returns:
            PDE loss tensor
        """
        u = self.model(self.X)
        
        # Compute first derivatives
        du_dX = torch.autograd.grad(
            u, self.X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        
        du_dx = du_dX[:, 0]
        du_dt = du_dX[:, 1]
        
        # Compute second derivative w.r.t. x
        du_dXX = torch.autograd.grad(
            du_dX, self.X,
            grad_outputs=torch.ones_like(du_dX),
            create_graph=True,
            retain_graph=True
        )[0]
        
        du_dxx = du_dXX[:, 0]
        
        # Burgers equation residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
        residual_left = du_dt + u.squeeze() * du_dx
        residual_right = (self.nu / math.pi) * du_dxx
        
        return self.criterion(residual_left, residual_right)
    
    def loss_function(self):
        """
        Compute total loss (data loss + physics loss).
        
        Returns:
            Total loss tensor
        """
        self.adam.zero_grad()
        self.lbfgs.zero_grad()
        
        # Data loss (boundary and initial conditions)
        y_pred = self.model(self.X_train)
        loss_data = self.criterion(y_pred, self.y_train)
        
        # Physics loss (PDE residual)
        loss_pde = self.compute_pde_loss()
        
        # Total loss
        loss = loss_data + loss_pde
        loss.backward()
        
        # Print progress
        if self.iter % 100 == 0:
            print(f"Iteration {self.iter:5d} | Total Loss: {loss.item():.6e} | "
                  f"Data Loss: {loss_data.item():.6e} | PDE Loss: {loss_pde.item():.6e}")
        
        self.iter += 1
        return loss
    
    def train(self, adam_iterations=1000):
        """
        Train the PINN using Adam followed by L-BFGS optimization.
        
        Args:
            adam_iterations: Number of Adam optimizer iterations
        """
        print("=" * 70)
        print("Training Physics-Informed Neural Network")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Adam iterations: {adam_iterations}")
        print("=" * 70)
        
        # Phase 1: Adam optimization
        print("\nPhase 1: Adam Optimization")
        print("-" * 70)
        self.model.train()
        for i in range(adam_iterations):
            self.adam.step(self.loss_function)
        
        # Phase 2: L-BFGS optimization
        print("\nPhase 2: L-BFGS Optimization")
        print("-" * 70)
        self.lbfgs.step(self.loss_function)
        
        print("\nTraining complete!")
    
    def predict(self, x, t):
        """
        Predict solution at given coordinates.
        
        Args:
            x: X coordinates (tensor or numpy array)
            t: T coordinates (tensor or numpy array)
        
        Returns:
            Predicted solution
        """
        self.model.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        
        X = torch.stack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T
        X = X.to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(X)
        
        return y_pred


def plot_solution(pinn, resolution=100, time_steps=None):
    """
    Plot the PINN solution.
    
    Args:
        pinn: Trained BurgersPINN instance
        resolution: Grid resolution for plotting
        time_steps: List of time step indices to plot in line plot
    """
    # Create high-resolution grid for prediction
    x = torch.linspace(-1, 1, resolution)
    t = torch.linspace(0, 1, resolution)
    
    X = torch.stack(torch.meshgrid(x, t, indexing='ij')).reshape(2, -1).T
    X = X.to(pinn.device)
    
    # Predict solution
    pinn.model.eval()
    with torch.no_grad():
        y_pred = pinn.model(X)
        y_pred = y_pred.reshape(len(x), len(t)).cpu().numpy()
    
    # Plot heatmap
    sns.set_style("white")
    plt.figure(figsize=(10, 6))
    sns.heatmap(y_pred, cmap='jet', cbar_kws={'label': 'u(x,t)'})
    plt.title("1D Burgers Equation Solution (PINN)", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Spatial Step", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Plot line plots at different times
    if time_steps is None:
        time_steps = [0, 33, 66, 99]
    
    plt.figure(figsize=(10, 6))
    for t_idx in time_steps:
        if t_idx < len(t):
            plt.plot(x.cpu().numpy(), y_pred[:, t_idx], 
                    label=f't = {t_idx / (resolution - 1):.2f}')
    
    plt.xlabel("x", fontsize=12)
    plt.ylabel("u(x,t)", fontsize=12)
    plt.title("1D Burgers Equation - Solution at Different Time Steps", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the PINN for 1D Burgers equation."""
    # Create and train PINN
    pinn = BurgersPINN(
        x_range=(-1, 1),
        t_range=(0, 1),
        h=0.1,
        k=0.1,
        nu=0.01
    )
    
    # Train the model
    pinn.train(adam_iterations=1000)
    
    # Plot results
    print("\nGenerating plots...")
    plot_solution(pinn, resolution=100, time_steps=[0, 33, 66, 99])


if __name__ == "__main__":
    main()
