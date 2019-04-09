from collections import namedtuple
import numpy as np

N_DIM = 2

CostModel = namedtuple('CostModel', [
    'callable', 'num_dimensions', 'parameters_boundaries'])

def assert_dimension(method):
    def wrapper(x, *args, **kwargs):
        assert len(x) == N_DIM
        return method(x, *args, **kwargs)
    return wrapper

# @assert_dimension
def _rastrigin(x, a=10):
    return len(x) * a + sum(np.square(x)  - a * np.cos(2 * np.pi * x))

# @assert_dimension
def _rosenbrock(x):
    s = 0
    for d in len(x):
        if not d % 2 == 0:
            s += np.square(1 - x[d - 1]) + 100 * np.square(x[d] - np.square(x[d-1]))
    return s

# @assert_dimension
def _ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(np.square(x)))) -\
        np.exp(0.5 * np.sum(np.cos(2 * np.pi * x))) + 20

RastriginModel = CostModel(callable=_rastrigin, num_dimensions=N_DIM, parameters_boundaries=[-5.12, 5.12])
RosenbrockModel = CostModel(callable=_rosenbrock, num_dimensions=N_DIM, parameters_boundaries=[-2.4, 2.4])
AckleyModel = CostModel(callable=_ackley, num_dimensions=N_DIM, parameters_boundaries=[-2.4, 2.4])
