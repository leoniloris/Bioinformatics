from pso.cost import RastriginModel
from pso.pso import PSO
import logging as log


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    pso = PSO(RastriginModel, num_particles=100, max_iter=1000, min_cummulative_error=1E-5)
    pso.optimize()
