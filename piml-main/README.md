# Physics-Informed Neural Networks (PINNs) Experiments

A comprehensive collection of well-documented, production-ready implementations demonstrating Physics-Informed Neural Networks (PINNs) for solving partial differential equations (PDEs). This repository contains both traditional numerical methods (Finite Difference Method) and modern machine learning approaches using PINNs and DeepXDE.

## Overview

Physics-Informed Neural Networks combine the power of deep learning with physical laws encoded as differential equations. This repository explores various implementations ranging from classical numerical solutions to state-of-the-art PINN architectures for solving fundamental PDEs in computational physics.

All implementations follow Python best practices with:
- Comprehensive docstrings
- Modular, reusable code
- Type hints where appropriate
- Clear separation of concerns
- Production-ready structure

## Repository Structure

```
PINN_experiments/
│
├── requirements.txt                        # Python dependencies
├── ref.txt                                 # Reference links and resources
│
├── 2 FDM Numerical Solution 1D Heat Equation/
│   └── fdm_heat_1d.py                     # Clean, documented FDM implementation
│
├── 3 FDM Numerical Solution for 2D Burgers Equation/
│   └── fdm_burgers_2d.py                  # 2D Burgers equation solver
│
├── 4 PINNs Solution for 1D Burgers Equation/
│   └── pinn_burgers_1d.py                 # PINN with PyTorch
│
├── 5 PINNs Solution for 2D Heat Equation/
│   └── pinn_heat_2d.py                    # 2D heat diffusion PINN
│
├── 6 DeepXDE Solution for 1D Heat/
│   ├── deepxde_heat_1d.py                 # DeepXDE implementation
│   ├── images/                            # Output visualizations
│   └── *.dat                              # Training logs
│
└── 7 DeepXDE Solution for 2D Navier Stokes/
    ├── deepxde_navier_stokes_2d.py        # Fluid dynamics with DeepXDE
    └── images/                            # Output visualizations
```

## Implementations

### 1. Finite Difference Method (FDM) Solutions

#### 1D Heat Equation (`fdm_heat_1d.py`)
- **Directory**: `2 FDM Numerical Solution 1D Heat Equation/`
- **Equation**: ∂u/∂t = k ∂²u/∂x²
- **Method**: Classical finite difference discretization
- **Description**: Numerical solution for heat diffusion in a 1D rod using explicit finite difference scheme
- **Features**:
  - Modular functions for grid initialization, PDE solving, and visualization
  - Configurable boundary conditions and material properties
  - Performance-optimized with precomputed constants

#### 2D Burgers Equation (`fdm_burgers_2d.py`)
- **Directory**: `3 FDM Numerical Solution for 2D Burgers Equation/`
- **Equations**: 
  - ∂u/∂t + u∂u/∂x + v∂u/∂y = ν(∂²u/∂x² + ∂²u/∂y²)
  - ∂v/∂t + u∂v/∂x + v∂v/∂y = ν(∂²v/∂x² + ∂²v/∂y²)
- **Method**: Finite difference method for nonlinear coupled PDEs
- **Description**: Numerical solution for the viscous Burgers equations in 2D with convection and diffusion
- **Features**:
  - Handles coupled velocity fields (u, v)
  - Progress tracking for long simulations
  - Contour plot visualizations

### 2. Physics-Informed Neural Networks (PINNs) Solutions

#### 1D Burgers Equation (`pinn_burgers_1d.py`)
- **Directory**: `4 PINNs Solution for 1D Burgers Equation/`
- **Framework**: PyTorch
- **Equation**: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
- **Architecture**: Deep neural network with Tanh activation [2→20→30→30→20→20→1]
- **Description**: PINN implementation using automatic differentiation to enforce PDE constraints
- **Features**:
  - Modular `BurgersNN` class for neural network architecture
  - `BurgersPINN` class handling training, loss computation, and prediction
  - Two-phase optimization: Adam followed by L-BFGS
  - Physics loss computed via automatic differentiation
  - Comprehensive visualization with heatmaps and line plots
- **Training**: ~1000 Adam iterations + L-BFGS refinement

#### 2D Heat Equation (`pinn_heat_2d.py`)
- **Directory**: `5 PINNs Solution for 2D Heat Equation/`
- **Framework**: PyTorch
- **Equation**: ∂u/∂t = α²(∂²u/∂x² + ∂²u/∂y²)
- **Description**: Extension of PINN approach to 2D heat diffusion problem
- **Features**:
  - Clean `HeatPINN` network class with flexible architecture
  - Automatic enforcement of initial and boundary conditions
  - Random collocation point generation for training
  - Solution visualization at multiple time steps
  - Loss tracking: IC loss + BC loss + PDE loss
- **Initial Condition**: u(x,y,0) = sin(πx)·sin(πy)
- **Boundary Conditions**: u = 0 at all boundaries

### 3. DeepXDE Solutions

[DeepXDE](https://deepxde.readthedocs.io/) is a library for scientific machine learning and physics-informed learning that provides high-level APIs for PINN development.

#### 1D Heat Equation (`deepxde_heat_1d.py`)
- **Directory**: `6 DeepXDE Solution for 1D Heat/`
- **Framework**: DeepXDE + TensorFlow
- **Equation**: ∂u/∂t = k·∂²u/∂x²
- **Description**: Using DeepXDE's high-level API to solve 1D heat equation with automated geometry handling
- **Features**:
  - Modular functions for PDE definition, geometry, and conditions
  - Automated PINN architecture with FNN [2→20→20→20→1]
  - Built-in boundary and initial condition specification
  - Training data logging (loss.dat, test.dat, train.dat)
  - Two-phase training: Adam (15k iterations) + L-BFGS
  - Visualization of meshless training point distribution
- **Initial Condition**: u(x,0) = sin(πx/L)
- **Boundary Conditions**: u = 0 at boundaries

#### 2D Navier-Stokes Equations (`deepxde_navier_stokes_2d.py`)
- **Directory**: `7 DeepXDE Solution for 2D Navier Stokes/`
- **Framework**: DeepXDE + TensorFlow
- **Equations**: Steady-state momentum + continuity
  - U·∂U/∂x + V·∂U/∂y = -(1/ρ)·∂P/∂x + ν·(∂²U/∂x² + ∂²U/∂y²)
  - U·∂V/∂x + V·∂V/∂y = -(1/ρ)·∂P/∂y + ν·(∂²V/∂x² + ∂²V/∂y²)
  - ∂U/∂x + ∂V/∂y = 0
- **Description**: Advanced fluid dynamics simulation solving coupled PDEs for velocity (U, V) and pressure (P) fields
- **Features**:
  - Complex boundary conditions: inlet, outlet, and wall boundaries
  - Network architecture: [2→64→64→64→64→64→3] with 5 hidden layers
  - Comprehensive visualization of velocity and pressure fields
  - Training with 10k Adam epochs + 3k L-BFGS iterations
  - Modular boundary condition functions
- **Complexity**: High - solving 3 coupled nonlinear PDEs simultaneously
- **Training Time**: ~20 minutes total (hardware dependent)

## Requirements

### Installation

Install all dependencies using the provided requirements file:

```bash
# Clone the repository
git clone https://github.com/yourusername/PINN_experiments.git
cd PINN_experiments

# Install dependencies
pip install -r requirements.txt
```

**Note**: PyTorch installation may vary based on your system. Visit [pytorch.org](https://pytorch.org) for system-specific installation instructions (CPU vs GPU/CUDA).

## Usage

All implementations are standalone Python scripts with a `main()` function. Simply run them directly:

### Finite Difference Method Examples

```bash
# 1D Heat Equation
cd "2 FDM Numerical Solution 1D Heat Equation"
python fdm_heat_1d.py

# 2D Burgers Equation
cd "../3 FDM Numerical Solution for 2D Burgers Equation"
python fdm_burgers_2d.py
```

### PINN Examples (PyTorch)

```bash
# 1D Burgers Equation with PINN
cd "4 PINNs Solution for 1D Burgers Equation"
python pinn_burgers_1d.py

# 2D Heat Equation with PINN
cd "../5 PINNs Solution for 2D Heat Equation"
python pinn_heat_2d.py
```

### DeepXDE Examples

```bash
# 1D Heat Equation with DeepXDE
cd "6 DeepXDE Solution for 1D Heat"
python deepxde_heat_1d.py

# 2D Navier-Stokes with DeepXDE (longer training time)
cd "../7 DeepXDE Solution for 2D Navier Stokes"
python deepxde_navier_stokes_2d.py
```

### Using as a Module

All implementations can be imported and used programmatically:

```python
from pinn_burgers_1d import BurgersPINN, plot_solution

# Create and train PINN
pinn = BurgersPINN(x_range=(-1, 1), t_range=(0, 1))
pinn.train(adam_iterations=1000)

# Generate predictions
plot_solution(pinn, resolution=100)
```

## Key Concepts

### Physics-Informed Neural Networks (PINNs)
PINNs are neural networks trained to solve supervised learning tasks while respecting physical laws described by PDEs. The loss function combines:
- **Data loss**: Fitting initial/boundary conditions
- **Physics loss**: Satisfying the governing PDE using automatic differentiation
- **Regularization**: Preventing overfitting

### Workflow for PINN Solutions
1. **Define Neural Network**: Architecture with appropriate input/output dimensions
2. **Set Conditions**: Initial conditions (IC) and boundary conditions (BC)
3. **Configure Optimizer**: Adam, L-BFGS, or other optimization algorithms
4. **Loss Function**: Combine data loss + physics loss
5. **Training Loop**: Iterative optimization
6. **Post-Processing**: Visualization and validation

## Mathematical Equations

### Heat Equation (1D)
```
∂u/∂t = k ∂²u/∂x²
```

### Burgers Equation (1D)
```
∂u/∂t + u∂u/∂x = ν ∂²u/∂x²
```

### Navier-Stokes Equations (2D)
```
∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
∂u/∂x + ∂v/∂y = 0
```

## References

### Learning Resources
- [DeepXDE Documentation](https://deepxde.readthedocs.io/en/latest/)
- [DeepXDE ICBC Module](https://deepxde.readthedocs.io/en/latest/modules/deepxde.icbc.html)
- [Kovasznay Flow Demo](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html)
- [Euler Beam Demo](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/eulerbeam.html)
- [I-Systems PINN Tutorial](https://i-systems.github.io/teaching/DL/iNotes_tf2/19_PINN_tf2.html)
- [ACE Numerics - Cavity Sessions](https://www.acenumerics.com/the-cavity-sessions.html)

### Code References
- [DeepXDE Wave Equation Example](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/wave_1d.py)
- [Machine Decision Code Samples](https://github.com/machinedecision/code_sample/tree/main)
- [Heat PDE Example](https://github.com/machinedecision/code_sample/blob/main/heat_pde.ipynb)
- [Burgers PyTorch Example](https://github.com/machinedecision/code_sample/blob/main/Burgers_torch.ipynb)

## Code Quality Features

This repository emphasizes production-ready code with:

### Documentation
- **Comprehensive docstrings**: Every function and class is documented with:
  - Purpose description
  - Parameter specifications with types
  - Return value descriptions
  - Usage examples where appropriate
- **Inline comments**: Complex algorithms include step-by-step explanations
- **Module-level documentation**: Each file includes an overview and equation definitions

### Code Structure
- **Modular design**: Functions are focused on single responsibilities
- **Reusable components**: Classes and functions can be imported and reused
- **Separation of concerns**: Clear distinction between model definition, training, and visualization
- **No code duplication**: Common patterns are abstracted into reusable functions

### Best Practices
- **Type consistency**: Proper tensor/array handling throughout
- **Performance optimization**: Precomputed constants, vectorized operations
- **Error handling**: Graceful device detection (CPU/GPU)
- **Clean main functions**: Easy to understand and modify
- **Progress tracking**: Informative training logs and iteration counters

## File Organization

- **`.py` files**: Clean, refactored Python implementations with proper structure
- **`images/`**: Visualization outputs and diagrams from DeepXDE runs
- **`*.dat` files**: Training data and loss history from DeepXDE (auto-generated)
- **`requirements.txt`**: All Python dependencies with version specifications
- **`ref.txt`**: Curated list of references and learning resources

## Contributing

This repository serves as a learning resource for physics-informed machine learning. Contributions are welcome:

- **New implementations**: Add solutions for additional PDEs
- **Performance improvements**: Optimize existing code
- **Documentation**: Enhance explanations and examples
- **Bug fixes**: Report and fix any issues
- **Testing**: Add unit tests for critical functions

Please ensure any contributions maintain the code quality standards established in this repository.

## License

MIT License - Educational resource for Physics-Informed Neural Networks.

---

**Repository Status**: Refactored and production-ready. All Jupyter notebooks have been converted to well-documented Python scripts following industry best practices.
