# 2D Quantum Particle in a Box Simulator

This Python project provides a real time simulation of a quantum particle confined in a two-dimensional box, solving the time-dependent Schrödinger equation. The numerical method employed is an iterative approximation of the Crank-Nicolson scheme, tailored to maintain stability and accuracy over long time simulations. Notably, the implementation is matrix-free, enhancing its suitability for real time simulation by minimizing memory footprint and maximizing computational efficiency (for a non-GPU python implementation).

## Features

- **Matrix-Free Implementation**: Directly computes the action of the Hamiltonian on a state vector without explicit matrix representation, optimizing memory usage and computational scalability.
- **Crank-Nicolson Method**: Uses an iterative approximation of this method to ensure numerical stability and accuracy.
- **Customizable Box Dimensions**: Users can specify the size, grid density, and potential profile of the 2D box.
- **Visualization**: Utilizes VisPy to display the simulation as it runs in real time.

## Getting Started

### Prerequisites

Ensure you have Python 3.12 or higher installed, along with the following packages:
- `numpy`
- `scipy`
- `VisPy`

### Example

To see an example, run the run_double_slit.py script. It is currently set up to perform the double slit diffraction experiment by firing an electron at two slits. Commenting out line 49 will remove the potential barrier between the slits, showing single slit diffraction.
Theoretically any arbitrary potential can be used. However, giving the potential array steep discontinuities can cause the iterative Crank-Nicolson approximator to take a very long time to converge.
