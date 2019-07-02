#!/usr/bin/env python
"""
Run simulations for multiple data draws (e.g., for uncertainty analyses).

Usage
-----

./run_uncertainty.py --draws 2000 --spawn 16 file1.yaml file2.yaml [...]

This will run 2000 simulations for each of the model specification files
(**plus** one simulation using the mean values) and distribute these
simulations over 16 cores.
"""

import argparse
import logging
import sys

import mslt_port.parallel as parallel


def get_parser():
    p = argparse.ArgumentParser(
        description='Run uncertainty analyses',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('-d', '--draws', type=int, default=5,
                   help='The number of draws for which to run simulations')

    p.add_argument('-s', '--spawn', type=int, default=1,
                   help='The number of processes to run in parallel')

    p.add_argument('model', type=str, nargs='+', metavar='model.yaml',
                   help='The model specification file(s)')

    return p


def main(args=None):
    logging.basicConfig(level=logging.INFO)

    parser = get_parser()
    options = parser.parse_args(args)

    num_draws = options.draws
    num_procs = options.spawn
    spec_files = options.model

    success = parallel.run_many(spec_files, num_draws, num_procs)

    exit_code = 0 if success else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
