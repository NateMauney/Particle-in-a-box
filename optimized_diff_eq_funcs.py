from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    V: np.ndarray
    dx: float
    dt: float
    hbar: float
    m: float


class NondimensionalizedParams:
    tolerance = 1e-10   # tolerance for Crank–Nicolson convergence

    def __init__(self, psi0, V, dx, dt, hbar, m):
        self.original = Params(V, dx, dt, hbar, m)
        self.psi0 = psi0
        # Nondimensionalization
        self.L = dx   # characteristic length (self.dx is not present because this sets it to 1.000000...)
        self.E = hbar ** 2 / (m * self.L ** 2)   # characteristic energy
        self.T = hbar / self.E   # characteristic time
        self.dt = dt / self.T
        self.V = V / self.E
        print(f'{100 * "="}\nParameters:')
        print(f'Original -----------> dx: {dx:.4e}, dt: {dt:.4e}, max(V) - min(V): {V.max() - V.min():.4e}')
        print(f'Nondimensionalized -> dx: {1:.4e}, dt: {self.dt:.4e}, '
              f'max(V) - min(V): {self.V.max() - self.V.min():.4e}')
        # CFL doesn't really apply in a simulation this nonlinear
        print(f"Courant-Friedrichs-Lewy (CFL) condition: {(1/2) * self.dt = :.4f} <= 1")
        print(f'{100 * "="}')

    def laplacian(self, wave_func):
        """ Calculate the laplacian of wave_func """
        result = wave_func * -4
        result[1:, :] += wave_func[:-1, :]  # top of kernel (y)
        result[:-1, :] += wave_func[1:, :]  # bottom of kernel (y)
        result[:, 1:] += wave_func[:, :-1]  # right of kernel (x)
        result[:, :-1] += wave_func[:, 1:]  # left of kernel (x)
        # enforce boundary conditions
        result[:1, :] = result[-1:, :] = result[:, :1] = result[:, -1:] = 0
        return result

    def hamiltonian(self, wave_func):
        """ Apply the Hamiltonian operator """
        return self.V * wave_func - 0.5 * self.laplacian(wave_func)

    def half_euler_step(self, wave_func):
        """ Apply the Hamiltonian operator """
        return -0.5j * self.dt * self.hamiltonian(wave_func)

    def energy_expectation(self, wave_func):
        """ Actual energy expectation, not nondimensional """
        self.normalize(wave_func)
        energy = np.real(np.sum(wave_func.conj() * self.hamiltonian(wave_func))) * self.E * self.L**2
        return energy * 6.241509074e+18   # J -> eV

    def normalize(self, wave_func):
        """ Normalizes wave_func according to original x and y coordinates, not nondimensional """
        wave_func /= np.sqrt(np.sum(np.abs(wave_func) ** 2)) * self.L
        return wave_func

    def max_normalize(self, wave_func):
        """ Not normalization in the quantum mechanical sense """
        wave_func /= max(wave_func.real.max(), wave_func.imag.max())   # just prevent overflows in long runs
        return wave_func

    def crank_nicolson_generator(self, psi0, steps_per_yield):
        """ Integrate Schrödinger equation via approximating the Crank-Nicolson method """
        psi = np.copy(psi0)
        # enforce boundary conditions
        psi[:1, :] = psi[-1:, :] = psi[:, :1] = psi[:, -1:] = 0
        self.max_normalize(psi)   # not actual normalization, just preventing overflows
        yield psi.astype(np.complex64, copy=True), 0

        i = -2
        while True:
            for _ in range(3000):
                i += 1
                # psi_next = psi
                forward_half = psi - (0.5j * self.dt) * self.hamiltonian(psi)
                psi_next = forward_half
                while True:
                    new_psi_next = forward_half - (0.5j * self.dt) * self.hamiltonian(psi_next)
                    # convergence check
                    if not np.any(np.abs(new_psi_next - psi_next) > np.abs(psi_next) * self.tolerance):
                        break
                    psi_next = new_psi_next
                psi = psi_next

                if i % steps_per_yield == 0:
                    yield psi.astype(np.complex64, copy=True), (i+2) * self.dt * self.T
            self.max_normalize(psi)

    def euler_method_generator(self, psi0, steps_per_yield):
        """ Integrate Schrödinger equation via Euler method """
        psi = np.copy(psi0)
        psi[:1, :] = psi[-1:, :] = psi[:, :1] = psi[:, -1:] = 0
        self.max_normalize(psi)   # not actual normalization, just preventing overflows
        yield np.copy(psi).astype(np.complex64), 0
        i = -2
        while True:
            i += 1
            psi = psi - (1j * self.dt) * self.hamiltonian(psi)
            # normalization condition
            psi = self.max_normalize(psi)
            # record the wave function
            if i % steps_per_yield == 0:
                yield psi.astype(np.complex64, copy=True), (i+2) * self.dt * self.T


def electron_plane_wave_function(X, Y, t, sigma_x, sigma_y, x_0, y_0, eV):
    """
    Calculate the 2D complex wave function of an electron on a meshgrid with specified initial positions x_0 and y_0.

    Parameters:
    - X, Y (ndarray): Meshgrid coordinates to evaluate the wave function.
    - t (float): Time at which to evaluate the wave function.
    - sigma_x, sigma_y (float): Initial spreads in the x and y directions (angstroms)
    - x_0, y_0 (float): Initial center positions of the wave packet in x and y directions (angstroms).
    - eV (float): Energy of the wave packet (electron volts)

    Returns:
    - ndarray: Complex values of the wave function at each point (x, y).
    """
    # angstroms -> meters
    sigma_x *= 1e-10
    sigma_y *= 1e-10
    x_0 *= 1e-10
    y_0 *= 1e-10

    # Constants
    m_e = 9.10938356e-31  # Mass of an electron (kg)
    hbar = 1.0545718e-34  # Reduced Planck's constant (Joule.second)
    energy_J = eV * 1.60218e-19  # Energy in Joules for 100 eV
    p = (2 * m_e * energy_J) ** 0.5
    k = p / hbar
    v = p / m_e
    omega = (hbar * k ** 2) / (2 * m_e)

    # Time-dependent spread
    sigma_x_t = sigma_x * np.sqrt(1 + (hbar * t / (2 * m_e * sigma_x ** 2)) ** 2)
    sigma_y_t = sigma_y * np.sqrt(1 + (hbar * t / (2 * m_e * sigma_y ** 2)) ** 2)

    # Amplitude of the wave function
    amplitude = (1 / (2 * np.pi * sigma_x_t * sigma_y_t)) ** 0.5

    # Phase of the wave function
    phase = k * X - omega * t  # Linear phase component

    # Wave function
    psi = amplitude * np.exp(
        -((X - x_0 - v * t) ** 2 / (2 * sigma_x_t ** 2) + (Y - y_0) ** 2 / (2 * sigma_y_t ** 2))) * np.exp(1j * phase)

    return psi
