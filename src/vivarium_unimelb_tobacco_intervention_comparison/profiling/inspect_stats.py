#!/usr/bin/env python
"""
Use this script to inspect the profiling results of a Vivarium simulation.
"""

import argparse
from pstats import Stats
import sys


def run_sim(sim_file):
    sim = setup_simulation_from_model_specification(sim_file)
    sim.run()
    sim.finalize()


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--calls',
                   action='append_const', dest='sortby', const='calls',
                   help='Sort by the number of calls')
    p.add_argument('--cumulative',
                   action='append_const', dest='sortby', const='cumulative',
                   help='Sort by cumulative time')
    p.add_argument('--time',
                   action='append_const', dest='sortby', const='time',
                   help='Sort by internal time')
    p.add_argument('--match', metavar='PATTERN',
                   action='append', dest='match',
                   help='Limit output to functions matching PATTERN')
    p.add_argument('--fraction', metavar='FRACTION',
                   action='append', dest='match', type=float,
                   help='Limit output to some FRACTION (0..1)')
    p.add_argument('stats_file',
                   metavar='specification_file.stats')
    return p


def main(args=None):
    parser = get_parser()
    opts = parser.parse_args(args)
    if opts.sortby is None:
        opts.sortby = ['cumulative', 'calls']
    if opts.match is None:
        opts.match = []
    print(opts.sortby)
    stats_file = opts.stats_file
    stats = Stats(stats_file)
    stats.sort_stats(*opts.sortby)
    try:
        stats.print_stats(*opts.match)
    except BrokenPipeError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
