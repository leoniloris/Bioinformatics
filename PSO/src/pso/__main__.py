from pso.cost import RastriginModel
from pso.pso import PSO
import logging as log
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('plot', action='store_true', help='plot the PSO evolution.')
    parser.add_argument('population', nargs='?', type=int, const=100)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    log.basicConfig(level=log.INFO)
    pso = PSO(
        RastriginModel,
        num_particles=100, max_iter=1000, min_cummulative_error=1E-5)
    states = pso.optimize()
    if args.plot: plot(states)
