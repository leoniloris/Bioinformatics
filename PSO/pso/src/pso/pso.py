from multiprocessing.pool import ThreadPool
from collections import defaultdict
from particle import Particle
from functools import partial
from functools import reduce

import logging as log
import numpy as np


# _pool = ThreadPool(16)

class PSO:
    def __init__(self, cost_model, num_particles, max_iter, min_cummulative_error):
        self._min_cummulative_error = min_cummulative_error
        self._max_iter = max_iter
        self._swarm_best_error = np.Inf
        self._cost_model = cost_model
        self._swarm_best_position = np.random.uniform(*cost_model.parameters_boundaries, size=cost_model.num_dimensions)
        self._swarm = [
            Particle(np.random.uniform(*cost_model.parameters_boundaries, size=cost_model.num_dimensions))
            for _ in range(num_particles)]
        self._states = defaultdict(list)

    def optimize(self):
        current_iteration = -1
        def _done():
            nonlocal current_iteration
            current_iteration += 1
            return any(
                (current_iteration >= self._max_iter,
                 abs(self._get_swarm_cummulative_error()) < self._min_cummulative_error))

        while not _done():
            self._evaluate_swarm_fitness()
            self._update_swarm_state()
            self._save_state()

        log.info('best position: %s\n error: %f' % (str(self._swarm_best_position), self._swarm_best_error))
        return self._states.copy()

    def _update_particle_fitness(self, particle):
        return particle.update_fitness(self._cost_model.callable)

    def _update_particle_state(self, particle):
        return particle.update_state(self._swarm_best_position, *self._cost_model.parameters_boundaries)

    def _evaluate_swarm_fitness(self):
        # _swarm_error = _pool.map(self._update_particle_fitness, self._swarm)
        _swarm_error = list(map(self._update_particle_fitness, self._swarm))
        _min_swarm_current_error = np.min(_swarm_error)
        if _min_swarm_current_error < self._swarm_best_error:
            self._swarm_best_position, self._swarm_best_error = \
                self._swarm[np.argmin(_swarm_error)].position, _min_swarm_current_error

    def _update_swarm_state(self):
        # _pool.map(self._update_particle_state, self._swarm)
        list(map(self._update_particle_state, self._swarm))

    def _get_swarm_cummulative_error(self):
        return reduce(lambda cummulated, particle: cummulated + particle.error, self._swarm, 0)

    def _save_state(self):
        self._states['cumulative_error'].append(self._get_swarm_cummulative_error())
        self._states['particles_positions'].append(
            {particle_number: list(particle.position) \
                for particle_number, particle in enumerate(self._swarm)}
        )
