#!/usr/bin/env python
"""
This script plots the survival curves of a specific cohort (in this case,
those aged 50-54 years at the start of the simulation).

USAGE:

    ./plot_survival_curves.py intervention1.yaml intervention2.yaml ...

"""

import argparse
import collections
import contextlib
from cycler import cycler
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.ticker
import numpy as np
import os.path
import pandas as pd
import re
import sys


def get_re_match(yaml_file, pattern, descr, group=1):
    regex = re.compile(pattern, flags=re.ASCII)

    with open(yaml_file, 'r', encoding='ascii') as f:
        for line in f:
            match = regex.match(line)
            if match is not None:
                return match.group(group)

    msg = 'could not identify {} in {}'
    raise ValueError(msg.format(descr, yaml_file))


def get_intervention_name(yaml_file):
    """
    Retrieve the intervention name from a simulation file.

    This is done by searching for a line that contains the pattern
    '- TobaccoXXX()' and extracting the class name.
    """
    pattern = r'^\s+-\s+(Tobacco\w+|Modify\w+)\(.*\)\s*$'
    interventions = {'TobaccoEradication': 'Eradication',
                     'TobaccoFreeGeneration': 'TFG',
                     'ModifyDiseaseIncidence': 'Reduce CHD 5%',
                     'ModifyAllCauseMortality': 'Reduce ACMR 5%'}

    intervention = get_re_match(yaml_file, pattern, 'intervention')
    if intervention in interventions:
        return interventions[intervention]

    msg = 'unknown intervention {} in {}'
    raise ValueError(msg.format(intervention, yaml_file))


def get_artifact_path(yaml_file):
    """
    Retrieve the filename of the input data artifact from a simulation file.
    """
    pattern = r'^\s+artifact_path:\s+(.*)\.hdf\s*$'
    return get_re_match(yaml_file, pattern, 'artifact path')


def get_population_name(yaml_file):
    """
    Retrieve the name of the population from a simulation file.

    This is done by identifying whether the artifact path contains '_maori_'
    or '_non-maori_'.
    """
    # populations = {'_maori_': 'Maori', '_non-maori_': 'non-Maori'}
    populations = {'_maori_': 'M', '_non-maori_': 'NM'}

    artifact_path = get_artifact_path(yaml_file)
    for population_key, population_name in populations.items():
        if population_key in artifact_path:
            return population_name

    msg = 'could not identify population in {}'
    raise ValueError(msg.format(yaml_file))


def get_tobacco_delay(yaml_file):
    pattern = r'^\s+artifact_path:.*_(\d+)-years\.hdf'
    delays = {'20': 20,
              '0': 0}

    delay_str = get_re_match(yaml_file, pattern, 'constant_prevalence')
    if delay_str in delays:
        return delays[delay_str]

    msg = 'could not identify delay in {}'
    raise ValueError(msg.format(yaml_file))


def get_bau_name(yaml_file):
    pattern = r'^\s+constant_prevalence:\s+(\w+)\s*$'
    baus = {'True': 'Constant Prevalence',
            'False': 'Decreasing Prevalence'}
    default_str = 'False'

    try:
        prev_str = get_re_match(yaml_file, pattern, 'constant_prevalence')
        if prev_str in baus:
            return baus[prev_str]
    except ValueError:
        # Ignore this, the default is 'False'.
        return baus[default_str]

    msg = 'could not identify BAU in {}'
    raise ValueError(msg.format(yaml_file))


def get_survival_data(popn_data, column):
    """
    Return a table with two columns, 'year' and 'pcnt', that describe the
    survival rate of a population.

    :param popn_data: The HALY table for the target cohort.
    :param column: Either 'population' (for the intervention), or
        'bau_population' (for the BAU).
    """
    data = popn_data.loc[:, ['year', column]]
    data = data.groupby('year').sum().reset_index().reset_index(drop=True)
    data['pcnt'] = 100 * data[column] / data.loc[0, column]
    return data


def collect_survival_data(yaml_files, age=50):
    """
    Assemble a nested dictionary of population survival data
    """
    bin_width = 5
    year_0 = 2010 - age - (bin_width - 1)
    years_of_birth = [year_0 + delta for delta in range(bin_width)]

    Data = collections.namedtuple('Data', 'bau delay popn interv df_bau df_int')
    plot_data = []

    for ix, yaml_file in enumerate(yaml_files):
        haly_file = re.sub('\\.yaml$', '_mm.csv', yaml_file)
        if not os.path.exists(haly_file):
            # NOTE: this may affect the consistency of the colours in each
            # sub-plot.
            print('WARNING: {} not found'.format(haly_file))
            continue
        haly_data = pd.read_csv(haly_file)

        # Select the target cohort.
        mask = haly_data['year_of_birth'].isin(years_of_birth)
        cohort_data = haly_data[mask]

        bau_data = get_survival_data(cohort_data, 'bau_population')
        interv_data = get_survival_data(cohort_data, 'population')

        bau_name = get_bau_name(yaml_file)
        # delay_years = get_tobacco_delay(yaml_file)
        delay_name = 'Recover in {} years'.format(get_tobacco_delay(yaml_file))
        popn_name = get_population_name(yaml_file)
        # bau_label = legend_fmt.format(popn_name, 'BAU')
        interv_name = get_intervention_name(yaml_file)
        # interv_label = legend_fmt.format(popn_name, interv_name)

        data = Data(bau=bau_name, delay=delay_name,
                    popn=popn_name, interv=interv_name,
                    df_bau = bau_data, df_int = interv_data)
        plot_data.append(data)

    return plot_data


def plot_survival_by(plot_data, row_fn, col_fn, label_fn, colour_map='Set2',
                     bau=True, interv=True):
    data_table = {}

    for data in plot_data:
        row_val = row_fn(data)
        col_val = col_fn(data)
        label = label_fn(data)

        if row_val not in data_table:
            data_table[row_val] = {}

        if col_val not in data_table[row_val]:
            data_table[row_val][col_val] = {}

        if label not in data_table[row_val][col_val]:
            data_table[row_val][col_val][label] = data
        else:
            raise ValueError('Duplicate label {} for {} and {}'.format(
                label, row_val, col_val))

    num_rows = len(data_table)
    num_cols = max(len(row_data) for row_data in data_table.values())

    num_colours = max(len(subplot_data) for row_data in data_table.values()
                      for subplot_data in row_data.values())
    # Each experiment may contribute three data series: BAU, intervention, and
    # the difference between the two.
    num_colours *= 3

    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True,
                           squeeze=False,
                           # Plot width and height, in inches.
                           figsize=(8, 6))
    cmap = plt.get_cmap(colour_map)
    colours = [cmap(ix) for ix in np.linspace(0, num_colours / 8.5, num_colours)]
    colour_cycler = cycler('color', colours)

    for rix, row_lbl in enumerate(sorted(data_table.keys())):
        for cix, col_lbl in enumerate(sorted(data_table[row_lbl].keys())):
            # For each scenario, plot all of the time series.
            plotted_bau = False
            ax[rix, cix].set_prop_cycle(colour_cycler)
            subplot_data = data_table[row_lbl][col_lbl]
            for label in sorted(subplot_data.keys()):
                df = subplot_data[label]
                if bau and not plotted_bau:
                    # ax[rix, cix].plot(df.df_bau['year'], df.df_bau['pcnt'],
                    #                   '--', label='BAU', color='black')
                    ax[rix, cix].plot(df.df_bau['year'], df.df_bau['pcnt'],
                                      label='BAU', linewidth=1)
                    plotted_bau = True
                if interv:
                    ax[rix, cix].plot(df.df_int['year'], df.df_int['pcnt'],
                                      label=label, linewidth=1)
                    df_bau = df.df_bau.loc[df.df_bau['year'] > 2010]
                    df_int = df.df_int.loc[df.df_int['year'] > 2010]
                    ax[rix, cix].plot(df_int['year'][1:],
                                      df_int['pcnt'][1:] - df_bau['pcnt'][1:],
                                      label='Survival Gain: ' + label,
                                      linewidth=1)

                # Only show x-axis labels in the bottom row.
                if rix == num_rows - 1:
                    ax[rix, cix].set_xlabel('Year')
                # Only show y-axis labels in the left column.
                if cix == 0:
                    ax[rix, cix].set_ylabel('Survival (%)')
                # Only show the legend in the top-right plot.
                if cix == num_cols - 1 and rix == 0:
                    ax[rix, cix].legend(loc='lower center')
                # Identify the scenario.
                title = '{} {}'.format(col_lbl, row_lbl)
                ax[rix, cix].set_title(title)
                ax[rix, cix].set_yscale('log')

                # Display y-axis labels as percentages.
                y_ticks = ax[rix, cix].get_yticks()
                y_ticks = ["{}%".format(y_tick) for y_tick in y_ticks]
                y_ticks = ax[rix, cix].set_yticklabels(y_ticks)

    return fig


@contextlib.contextmanager
def no_style_context():
    yield None


def get_parser():
    """The parser for command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('simulation_file', nargs='+')
    return p


def main(args=None):
    """The script entry point."""
    parser = get_parser()
    opts = parser.parse_args(args)

    plot_file = 'survival_curves.pdf'

    # See the following matplotlib style gallery:
    # https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
    plot_style = 'seaborn-white'

    # plot_ages = [50, 30, 10, 0]
    plot_ages = [50]
    plot_data = {age: collect_survival_data(opts.simulation_file, age=age)
                 for age in plot_ages}

    if plot_style in matplotlib.style.available:
        plot_context = matplotlib.style.context(plot_style, after_reset=True)
    else:
        print('Warning: plot style {} not supported'.format(plot_style))
        plot_context = no_style_context()

    plot_list = []
    with plot_context:
        # Plot the absolute survival curves for the BAU and the intervention.
        for age in plot_ages:
            plot = plot_survival_by(
                plot_data[age],
                lambda row: '{}, {}'.format(row.bau, row.delay),
                lambda col: '',
                lambda lbl: '{}'.format(lbl.interv))
            plot.suptitle('{}-{} year olds'.format(age, age + 4))
            plot_list.append(plot)

    print('Saving to {} ...'.format(plot_file))
    pages = matplotlib.backends.backend_pdf.PdfPages(plot_file)
    for plot in plot_list:
        pages.savefig(plot, bbox_inches='tight')
    pages.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
