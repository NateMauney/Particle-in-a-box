import time
import numpy as np
from PyQt5 import QtWidgets, QtCore
from dataclasses import dataclass
from vispy.app import use_app
import multiprocessing as mp
import queue
from vispy import scene
from vispy.color import get_colormap
from scipy.interpolate import RegularGridInterpolator
from collections import deque

def meshgrid_from_nx_ny(nx, ny, multiplier=1):
    a = 280
    b = a * ny / nx
    x = np.linspace(-a, a, int(np.round(nx * multiplier)), dtype=np.float32)
    y = np.linspace(-b, b, int(np.round(ny * multiplier)), dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


class CanvasWrapper(QtCore.QObject):
    closing = QtCore.pyqtSignal()

    def __init__(self, queue, params, relative_z_scale, title='Wave Func', size=(1920, 1080)):
        super().__init__()
        self.queue = queue
        self.params = params
        self.potential = params.V.astype(np.float32)
        self.relative_z_scale = relative_z_scale
        self.canvas = scene.SceneCanvas(title=title, keys='interactive', size=size)
        self.legend = self.canvas.central_widget.add_view()
        self.surface = self.canvas.central_widget.add_view()

        # set up grid for plotting (original size)
        ny, nx = params.V.shape
        self.x, self.y, _, _ = meshgrid_from_nx_ny(nx, ny)
        _, _, self.X, self.Y = meshgrid_from_nx_ny(nx, ny, self.params.plotting_params.interpolate_multiplier)

        self.x_diff, self.y_diff = abs(self.x.max() - self.x.min()), abs(self.y.max() - self.y.min())
        self.avg_diff = (self.x_diff + self.y_diff) / 2

        # set up camera
        self.surface.camera = scene.TurntableCamera(up='z', fov=60, elevation=90.0, azimuth=0.0)
        self.surface.camera.aspect = 'equal'
        self.surface.camera.center = (0, 0, 0)
        distance = max(self.x_diff, self.y_diff) / (2 * np.tan(np.radians(self.surface.camera.fov / 2)))
        self.surface.camera.distance = distance

        # set up a data structure to keep a short history of the frame times
        self.frame_count = 0
        self.num_frames_for_fps = 20
        self.frame_times = deque(np.full(self.num_frames_for_fps, time.perf_counter()), maxlen=self.num_frames_for_fps)

        # initialize the surface plot
        self.wave_func_surface = scene.visuals.SurfacePlot(x=self.X, y=self.Y, z=np.zeros_like(self.X))
        self.surface.add(self.wave_func_surface)

        # set up the potential
        self.get_potential_surfaceplot()

        # initialize the legend
        energy = 0
        #     f'Energy (eV): {energy}\nTime: {time:.3e}\nFrame: {0}\nFrame Rate: {0}', color='white', pos=(10, 10),
        #     anchor_x='left', anchor_y='bottom')
        self.legend_text = scene.visuals.Text(
            f'Energy (eV): {0}\n'
            f'Simulation Time: {0:.3e} s\n'
            f'Wave Function Size: [{self.potential.shape[0]}, {self.potential.shape[1]}]\n'
            f'Frame Rate: {0}\n'
            f'Frame: {0}',
            color='white', pos=(10, 10), anchor_x='left', anchor_y='bottom', method='gpu'
        )
        self.legend.add(self.legend_text)

        # make axes
        axis_margin = 0.02  # Adjust this value as needed
        start_pos = [self.x.min() - self.x_diff * axis_margin, self.y.min() - self.x_diff * axis_margin]
        font_size = 10 * self.avg_diff
        axis_label_margin = 100 * self.avg_diff
        x_range = self.params.original.dx * nx * 1e10
        y_range = self.params.original.dx * ny * 1e10

        minor_tick_length = 10 * self.avg_diff
        major_tick_length = minor_tick_length * 2
        tick_label_margin = 15 * self.avg_diff
        xax = scene.Axis(
            pos=[start_pos, [self.x.max(), start_pos[1]]], tick_direction=(0, -1), domain=(-x_range/2, x_range/2),
            font_size=font_size, axis_color='w', tick_color='w', text_color='w', parent=self.surface.scene,
            minor_tick_length=minor_tick_length, major_tick_length=major_tick_length,
            tick_label_margin=tick_label_margin, axis_label='X (Å)', axis_font_size=font_size,
            axis_label_margin=axis_label_margin
        )

        yax = scene.Axis(
            pos=[start_pos, [start_pos[0], self.y.max()]], tick_direction=(-1, 0), domain=(-y_range/2, y_range/2),
            font_size=font_size, axis_color='w', tick_color='w', text_color='w', parent=self.surface.scene,
            minor_tick_length=minor_tick_length, major_tick_length=major_tick_length,
            tick_label_margin=tick_label_margin, axis_label='Y (Å)', axis_font_size=font_size,
            axis_label_margin=axis_label_margin
        )

    def get_potential_surfaceplot(self, multiplier=3, scale=1.0):
        potential = self.potential
        potential_diff = potential.max() - potential.min()
        if potential_diff == 0:
            return
        potential *= self.avg_diff * self.relative_z_scale * scale / potential_diff

        ny, nx = self.params.V.shape
        x, y, _, _ = meshgrid_from_nx_ny(nx, ny)
        potential_interp = RegularGridInterpolator((y, x), potential, method='nearest')
        _, _, X, Y = meshgrid_from_nx_ny(nx, ny, multiplier=multiplier)
        potential = potential_interp((Y, X))
        colors = np.full((potential.shape[0] * potential.shape[1], 4), 0.5)
        colors[:, 3] = 0.3
        colors[:, 3] *= (potential != 0).flatten()
        up = scene.visuals.SurfacePlot(x=X, y=Y, z=potential)
        up.mesh_data.set_vertex_colors(colors)
        self.surface.add(up)
        return

    def update_data(self):
        self.frame_count += 1
        wave_func, colors, energy, sim_time = self.queue.get()
        self.wave_func_surface.set_data(z=wave_func)
        self.wave_func_surface.mesh_data.set_vertex_colors(colors)

        # find average frame rate over the last 'self.frame_times.maxlen' number of frames
        current_time = time.perf_counter()
        frame_rate = self.frame_times.maxlen / (current_time - self.frame_times.popleft())
        self.frame_times.append(current_time)

        self.legend_text.text = (f'Energy (eV): {energy}\n'
                                 f'Simulation Time: {sim_time:.3e} s\n'
                                 f'Wave Function Size: [{self.potential.shape[0]}, {self.potential.shape[1]}]\n'
                                 f'Frame Rate: {frame_rate:.1f}\n'
                                 f'Frame: {self.frame_count}'
                                 )

        self.canvas.update()

    def on_close(self, event):
        print("Canvas is closing!")
        self.closing.emit()  # Emit the close signal
        self.canvas.on_close(event)
        # super(SceneCanvas, self.canvas).on_close(event)


class MyMainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()

    def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()
        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)


class DataSource:
    """Object representing a complex data producer."""
    viridis_cmap = get_colormap('viridis')
    hsv_cmap = get_colormap('hsv')

    def __init__(self, out_queue, stop_data_source, initial_wave_func, params):
        self.out_queue = out_queue
        self.stop_data_source = stop_data_source
        self.wave_func_gen = params.crank_nicolson_generator(initial_wave_func, 1)

    def start_data_creation(self):
        print("DataSource is starting up")
        while True:
            # next_wave_func = self.get_next_wave_func()
            next_wave_func = next(self.wave_func_gen)
            successful_transfer = False
            while not successful_transfer:
                if self.stop_data_source.is_set():
                    break
                try:
                    self.out_queue.put(next_wave_func, timeout=0.3)
                    successful_transfer = True
                except queue.Full:
                    continue
            if self.stop_data_source.is_set():
                break
        print("DataSource has stopped")


class Relay(QtCore.QObject):
    """Object representing a complex data producer."""
    new_data = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self, in_queue, out_queue, stop_preprocessor, parent=None):
        super().__init__(parent)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_preprocessor = stop_preprocessor
        self._should_end = False

    def relay(self):
        print("Relay is starting up")
        while True:
            if self._should_end:
                print("relay got the signal to stop, passing it on to stop_preprocessor")
                self.stop_preprocessor.set()
                break
            try:
                self.out_queue.put(self.in_queue.get(timeout=0.3), timeout=0.3)
                self.new_data.emit()
            except (queue.Empty, queue.Full):
                continue

        self.finished.emit()
        print('Relay has stopped')

    def stop_data(self):
        print("Relay is stopping...")
        self._should_end = True


class PreProcessor:
    viridis_cmap = get_colormap('viridis')
    hsv_cmap = get_colormap('hsv')

    def __init__(self, in_queue, out_queue, stop_preprocessor, stop_data_source, params):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.stop_preprocessor = stop_preprocessor
        self.stop_data_source = stop_data_source
        self.params = params
        self.plotting_params = params.plotting_params
        plot = self.plotting_params.plot
        self.process_next_wave_func = self.get_next_wave_func_real if plot == 'real' else self.get_next_wave_func_amp

        ny, nx = self.params.V.shape
        self.x, self.y, _, _ = meshgrid_from_nx_ny(nx, ny)
        _, _, self.X, self.Y = meshgrid_from_nx_ny(nx, ny, multiplier=self.plotting_params.interpolate_multiplier)

        self.x_diff, self.y_diff = abs(self.x.max() - self.x.min()), abs(self.y.max() - self.y.min())
        self.avg_diff = (self.x_diff + self.y_diff) / 2

    def start(self):
        print("PreProcessor is starting up")
        while not self.stop_preprocessor.is_set():
            while not self.stop_preprocessor.is_set():
                try:
                    to_be_processed = self.in_queue.get(timeout=0.3)
                    break
                except queue.Empty:
                    continue
            if not self.stop_preprocessor.is_set():
                to_plot = self.preprocess(to_be_processed)
            while not self.stop_preprocessor.is_set():
                try:
                    self.out_queue.put(to_plot, timeout=0.3)
                    break
                except queue.Full:
                    continue

        print("PreProcessor got the signal to stop, passing it on to data source")
        self.stop_data_source.set()
        print('PreProcessor has stopped')

    def preprocess(self, next_items):
        wave_func, sim_time = next_items[0], next_items[1]
        wave, colors, energy = self.process_next_wave_func(wave_func)
        return wave, colors, energy, sim_time

    def get_next_wave_func_amp(self, wave_func):
        energy = self.params.energy_expectation(wave_func)
        # get next wave function and scale it so it stays in the figure
        wave_func = self.params.normalize(wave_func)
        wave_func *= 3e-7


        amplitude = np.abs(wave_func).astype(np.float32)
        # amplitude *= self.avg_diff * self.plotting_params.relative_z_scale / (amplitude.max() - amplitude.min())
        amplitude_interp = RegularGridInterpolator((self.y, self.x), amplitude, method='linear')
        amplitude = amplitude_interp((self.Y, self.X))
        return amplitude, self.viridis_cmap.map((amplitude / amplitude.max()).flatten()), energy

    def get_next_wave_func_real(self, wave_func):
        # get next wave function and scale it so it stays in the figure
        energy = self.params.energy_expectation(wave_func)
        # wave_func = upscale_matrix(wave_func)
        wave_func.real *= self.avg_diff * self.plotting_params.relative_z_scale / (wave_func.real.max() - wave_func.real.min())
        wave_func.imag /= wave_func.imag.max() * 2
        wave_func.imag += 0.5
        real_wave, imag_wave = wave_func.real.astype(np.float32), wave_func.imag.astype(np.float32)
        real_wave_interp = RegularGridInterpolator((self.y, self.x), real_wave, method='linear')
        imag_wave_interp = RegularGridInterpolator((self.y, self.x), imag_wave, method='linear')
        real_wave = real_wave_interp((self.Y, self.X))
        imag_wave = imag_wave_interp((self.Y, self.X))
        return real_wave, self.viridis_cmap.map(imag_wave.flatten()), energy


def create_and_start_data_source(source_relay_queue, stop_data_source, wave_func_gen, params):
    data_source = DataSource(source_relay_queue, stop_data_source, wave_func_gen, params)
    data_source.start_data_creation()

def create_and_start_preprocessor(in_queue, out_queue, stop_preprocessor, stop_data_source, params):
    data_source = PreProcessor(in_queue, out_queue, stop_preprocessor, stop_data_source, params)
    data_source.start()

@dataclass
class PlottingParams:
    relative_z_scale: float
    interpolate_multiplier: float
    interpolation_method: str
    plot: str


def upscale_matrix(data):
    """ Fast and dirty linear interpolation [N x M] -> [2N-1 x 2M-1] """
    rows, cols = data.shape
    new_rows, new_cols = 2 * rows - 1, 2 * cols - 1
    new_data = np.zeros((new_rows, new_cols), dtype=data.dtype)
    # fill in the original points
    new_data[0::2, 0::2] = data
    # interpolate in columns (for intermediate x values)
    new_data[0::2, 1::2] = (new_data[0::2, :-2:2] + new_data[0::2, 2::2]) / 2
    # interpolate in rows (for intermediate y values)
    new_data[1::2, :] = (new_data[:-2:2, :] + new_data[2::2, :]) / 2
    return new_data


def upscale_matrix_nearest(data):
    """ Nearest neighbor interpolation to match 'upscale_matrix' [N x M] -> [2N-1 x 2M-1] """
    rows, cols = data.shape
    new_rows, new_cols = 2 * rows - 1, 2 * cols - 1
    # create an index grid for rows and columns
    r_idx = np.repeat(np.arange(rows), 2)[:new_rows]
    c_idx = np.repeat(np.arange(cols), 2)[:new_cols]
    # create the new matrix
    new_data = data[r_idx[:, None], c_idx]
    return new_data


def start_plotting(initial_wave_func, params, relative_z_scale=0.1, interpolate_multiplier=1.,
                 interpolation_method='linear', update_interval=0.01, plot='real'):

    # give params a new attribute for all the plotting parameters
    params.plotting_params = PlottingParams(relative_z_scale, interpolate_multiplier, interpolation_method, plot)
    app = use_app("pyqt5")
    app.create()
    source_relay_queue = mp.Queue(maxsize=30)
    source_preprocessor_queue = mp.Queue(maxsize=30)
    preprocessor_relay_queue = mp.Queue(maxsize=30)

    relay_canvas_queue = queue.Queue(maxsize=30)
    stop_data_source = mp.Event()
    stop_preprocessor = mp.Event()

    # make a relay to relay information form data_source to canvas
    relay = Relay(preprocessor_relay_queue, relay_canvas_queue, stop_preprocessor)
    # put the relay in its own thread and tell it to start relaying information what its thread starts
    relay_thread = QtCore.QThread()
    relay.moveToThread(relay_thread)
    relay_thread.started.connect(relay.relay)

    canvas_wrapper = CanvasWrapper(relay_canvas_queue, params, relative_z_scale)
    canvas_wrapper.canvas.connect(canvas_wrapper.canvas.on_close)
    win = MyMainWindow(canvas_wrapper)

    # have the relay update the canvas when it sends something to be rendered
    relay.new_data.connect(canvas_wrapper.update_data)
    relay.finished.connect(relay_thread.quit, QtCore.Qt.DirectConnection)

    # make a data generation process
    data_source_process = mp.Process(
        target=create_and_start_data_source,
        args=(source_preprocessor_queue, stop_data_source, initial_wave_func, params)
    )

    # make a preprocessor process
    preprocessor_process = mp.Process(
        target=create_and_start_preprocessor,
        args=(source_preprocessor_queue, preprocessor_relay_queue, stop_preprocessor, stop_data_source, params)
    )

    # if the window is closed, tell the data source to stop
    win.closing.connect(relay.stop_data, QtCore.Qt.DirectConnection)

    # start the processes
    data_source_process.start()
    preprocessor_process.start()
    relay_thread.start()
    win.show()

    # start rendering
    app.run()

    # make sure the processes exited gracefully
    print("Waiting for PreProcessor to close gracefully...")
    data_source_process.join(timeout=10)
    if data_source_process.is_alive():
        print("preprocessor_process is still running...forcing termination")
        data_source_process.terminate()
    print("Waiting for Relay to close gracefully...")
    relay_thread.wait(5000)
    print("Waiting for DataSource to close gracefully...")
    data_source_process.join(timeout=10)
    if data_source_process.is_alive():
        print("data_source_process is still running...forcing termination")
        data_source_process.terminate()

    relay_thread.wait(5000)
