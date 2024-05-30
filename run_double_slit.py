import optimized_diff_eq_funcs as func
from vispy_funcs import start_plotting
import numpy as np


if __name__ == "__main__":
    l_x = 750.0e-10
    l_y = 500.0e-10
    nx = 450
    dx = l_x / nx
    ny = int(np.round(l_y / dx))

    x = np.arange(-l_x / 2, l_x / 2, dx)
    y = np.arange(-l_y / 2, l_y / 2, dx)
    X, Y = np.meshgrid(x, y)

    print(f'{l_x = }, {l_y = }, {nx = }, {ny = }, {dx = }')

    # dt = 8e-17
    dt = 4e-17
    m = 9.10938356e-31
    hbar = 1.0545718e-34

    # define the potential arbitrarily
    V = np.zeros_like(X)
    x_start = 198
    delta_x = 4
    y_center_width = 25
    y_side_width = 75
    slit_thickness = 10e-10
    slit_width = 10e-10
    slit_distance = 35e-10
    wall_length = l_x / 2
    wall_width = 100e-10
    height_eV = 2

    slit_inner_edge = slit_distance / 2 - slit_width / 2
    slit_outer_edge = slit_distance / 2 + slit_width / 2
    to_y_indices = lambda distance: int(np.round(distance * ny / l_y))
    to_x_indices = lambda distance: int(np.round(distance * nx / l_x))
    slit_inner_edge = to_y_indices(slit_inner_edge)
    slit_outer_edge = to_y_indices(slit_outer_edge)
    wall_length = to_x_indices(wall_length)
    wall_width = to_y_indices(wall_width)
    slit_thickness = to_x_indices(slit_thickness)

    V[0:ny//2-slit_outer_edge, wall_length:wall_length+slit_thickness] = height_eV * 1.60218e-19
    V[ny//2+slit_outer_edge:ny, wall_length:wall_length+slit_thickness] = height_eV * 1.60218e-19
    V[ny//2-slit_inner_edge:ny//2+slit_inner_edge, wall_length:wall_length+slit_thickness] = height_eV * 1.60218e-19
    # V[0:wall_width, 0:wall_length] = height_eV * 1.60218e-19
    # V[ny-wall_width:ny, 0:wall_length] = height_eV * 1.60218e-19

    # make initial wave functon
    # psi0 = func.electron_plane_wave_function(X, Y, 0, 40, 40, -220, 0, 0.6)
    # psi0 = func.electron_plane_wave_function(X, Y, 0, 40, 40, -220, 0, 1.5)
    psi0 = func.electron_plane_wave_function(X, Y, 0, 40, 40, -180, 0, 1.7)

    params = func.NondimensionalizedParams(psi0, V, dx, dt, hbar, m)
    start_plotting(psi0, params, relative_z_scale=0.15, interpolate_multiplier=1.2,
                   interpolation_method='linear', update_interval=0.1, plot='r eal')

