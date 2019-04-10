import numpy as np
import logging as log

class Particle:
    def __init__(self, inital_position):
        self._min_error = np.Inf
        self._n_dimensions = len(inital_position)
        self._position = inital_position.copy()
        self._velocity = np.random.uniform(-1.0, 1.0, size=len(inital_position))
        self._error = np.Inf
        self._best_position = np.zeros_like(self._position)

    def update_fitness(self, cost_func):
        self._error = cost_func(self._position)
        if self._error < self._min_error:
            self._best_position, self._min_error = self._position, self._error
        return self._error

    def update_velocity(self, global_best_position):
        inertia = 0.9
        cognitive_constant = 2
        social_constant = 1

        r1 = 0.5 * (np.random.uniform(size=self._n_dimensions) + 1)
        r2 = 0.5 * (np.random.uniform(size=self._n_dimensions) + 1)

        cognitive_velocity = cognitive_constant * r1 * (self._best_position - self._position)
        social_velocity = social_constant * r2 * (global_best_position - self._position)
        self._velocity = inertia * self._velocity + cognitive_velocity + social_velocity

    def update_position(self, min_position=-100, max_position=100):
        self._position = self._position + self._velocity
        np.clip(self._position, min_position, max_position)

    def update_state(self, global_best_position, min_position=-100, max_position=100):
        # log.info('Current velocity %s\nCurrent position %s' % (str(self._velocity), str(self._position)))
        self.update_velocity(global_best_position)
        self.update_position(min_position, max_position)

    @property
    def position(self):
        return self._position

    @property
    def error(self):
        return self._error
