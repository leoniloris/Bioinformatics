import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

plot_limits = [0, 0, 0, 0]

def _plot_background(cost_model):
    global plot_limits
    _, axs = plt.subplots(2, 1, figsize=(7, 14))

    n_rows = 150
    x = np.linspace(*cost_model.parameters_boundaries, num=n_rows)
    X, Y = np.meshgrid(x, x)

    arr = np.array([
            [cost_model.callable(np.array([X[row, col], Y[row, col]])) for col in range(n_rows)]
            for row in range(n_rows)
        ])
    plot_limits = [x[0], x[-1], x[0], x[-1]]
    axs[0].imshow(arr, extent=plot_limits)

    return axs[0].twinx(), axs[1]

def _update_particles(particles_positions, particle_axis):
    plt.sca(particle_axis)
    plt.cla()
    for _, particle_positions in particles_positions.items():
        plt.plot(particle_positions[0], particle_positions[1], '*')
        plt.axis(plot_limits)
    plt.pause(0.1)

def _update_loss(cumulative_error, loss_axis, idx):
    plt.sca(loss_axis)
    plt.plot(idx, cumulative_error, 'r*')
    plt.grid()

def _animate_particles(states, particle_axis, loss_axis, boundaries):
    for idx in tqdm(range(len(states['particles_positions']))):
        _update_particles(states['particles_positions'][idx], particle_axis)
        _update_loss(states['cumulative_error'][idx], loss_axis, idx)
        plt.show()

def plot_states(states, cost_model):
    plt.ion()
    particle_axis, loss_axis = _plot_background(cost_model)
    _animate_particles(states, particle_axis, loss_axis, cost_model.parameters_boundaries)
