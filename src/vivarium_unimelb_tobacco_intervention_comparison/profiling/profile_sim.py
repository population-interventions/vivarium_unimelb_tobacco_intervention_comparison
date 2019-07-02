#!/usr/bin/env python
"""
Use this script to profile a Vivarium simulation.
"""

import argparse
import cProfile
import sys

from vivarium.interface import setup_simulation_from_model_specification


def run_sim(sim_file):
    sim = setup_simulation_from_model_specification(sim_file)
    sim.run()
    sim.finalize()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('specification_file',
                   metavar='specification_file.yaml')
    return p


def main(args=None):
    parser = get_parser()
    opts = parser.parse_args(args)
    sim_file = opts.specification_file
    command = 'run_sim("{}")'.format(sim_file)
    out_file = sim_file.replace('.yaml', '.stats')
    cProfile.run(command, filename=out_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
