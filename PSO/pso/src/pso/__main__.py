from cost import RastriginModel, RosenbrockModel, AckleyModel
from plot import plot_states
from pso import PSO
import logging as log
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='plot the PSO evolution.')
    parser.add_argument('--population', nargs='?', type=int, const=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log.basicConfig(level=log.INFO)
    cost_model = AckleyModel
    pso = PSO(
        cost_model,
        num_particles=args.population, max_iter=1000, min_cummulative_error=1E-4)
    states = pso.optimize()
    if args.plot: plot_states(states, cost_model)
