#!/usr/bin/env python3

import logging
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings


def find_files(delay, tob_prev, interv, expected=False):
    template = 'mslt_tobacco_.*_{}-years_{}_{}_mm{}\\.csv'
    if expected:
        exp_pat = ''
    else:
        exp_pat = '_.+'
    rexp = re.compile(template.format(delay, tob_prev, interv, exp_pat))

    return [filename for filename in os.listdir('.') if rexp.match(filename)]


def summarise_results(delay, tob_prev, interv, verbose=False):
    log = logging.getLogger(__name__)

    file_list = find_files(delay, tob_prev, interv, expected=False)

    # NOTE: for LYs and HALYs, report data for:
    # 1. All cohorts combined (Maori and non-Maori).
    # 2. Maori females.
    # 3. Non-Maori males.

    # For ACMR and YLDR, report data for:
    # 1. Maori females aged 62 in 2041 and 2061; and
    # 2. Non-Maori males aged 62 in 2041 and 2061.
    rate_years = [2041, 2061]
    rate_ages  = [62]
    rate_cols = ['year', 'age', 'sex']

    dfs_LY = []
    dfs_ACMR = []
    dfs_YLDR = []

    draw_rexp = re.compile('^.*_([0-9]+)\\.csv$')

    for filename in file_list:
        if verbose:
            print('Loading {} ...'.format(filename))

        is_non_maori = '_non-maori_' in filename
        if is_non_maori:
            popn = 'non-maori'
            rate_sex = 'male'
        else:
            popn = 'maori'
            rate_sex = 'female'

        draw_match = draw_rexp.match(filename)
        draw_number = int(draw_match.group(1))

        df_in = pd.read_csv(filename)
        df_in = df_in.rename(columns={
            'person_years': 'LY',
            'bau_person_years': 'bau_LY',
        })

        df_in_acmr = df_in.loc[
            (df_in['year'].isin(rate_years))
            & (df_in['age'].isin(rate_ages))
            & (df_in['sex'] == rate_sex),
            rate_cols + ['bau_acmr', 'acmr']
        ]
        df_in_acmr['popn'] = popn
        dfs_ACMR.append(df_in_acmr)

        df_in_yldr = df_in.loc[
            (df_in['year'].isin(rate_years))
            & (df_in['age'].isin(rate_ages))
            & (df_in['sex'] == rate_sex),
            rate_cols + ['bau_yld_rate', 'yld_rate']
        ]
        df_in_yldr['popn'] = popn
        dfs_YLDR.append(df_in_yldr)

        ly_cols = ['bau_LY', 'LY', 'bau_HALY', 'HALY']
        totals = df_in.loc[:, ly_cols].sum(axis=0)
        male_totals = df_in.loc[df_in['sex'] == 'male',
                                ly_cols].sum(axis=0)
        female_totals = df_in.loc[df_in['sex'] == 'female',
                                  ly_cols].sum(axis=0)
        totals['sex'] = 'All'
        male_totals['sex'] = 'male'
        female_totals['sex'] = 'female'
        df_in_ly = pd.DataFrame.from_dict(
            {0: totals, 1: male_totals, 2: female_totals},
            orient='index',
            columns=totals.index)
        df_in_ly['popn'] = popn
        df_in_ly['draw_number'] = draw_number
        dfs_LY.append(df_in_ly)

    logging.info('Read data from {} files'.format(len(file_list)))

    # Combine the results from each simulation.
    df_LY = pd.concat(dfs_LY, ignore_index=True, sort=False)
    df_ACMR = pd.concat(dfs_ACMR, ignore_index=True, sort=False)
    df_YLDR = pd.concat(dfs_YLDR, ignore_index=True, sort=False)

    # Calculate the net LYs and HALYs for the total population (Maori and
    # non-Maori) by pairing simulations based on their draw number.
    df_all_popn = df_LY.loc[df_LY['sex'] == 'All']
    if len(df_all_popn['popn'].unique()) > 1:
        # NOTE: until all simulations have completed, only retain those where
        # we have results for both populations.
        mask_m = df_all_popn['popn'] == 'maori'
        mask_nm = df_all_popn['popn'] == 'non-maori'
        draws_m = set(df_all_popn.loc[mask_m, 'draw_number'].unique())
        draws_nm = set(df_all_popn.loc[mask_nm, 'draw_number'].unique())
        draws = list(draws_m & draws_nm)
        df_all_popn = df_all_popn.loc[df_all_popn['draw_number'].isin(draws)]

        df_groups = df_all_popn.groupby(['sex', 'draw_number'], as_index=False)
        df_totals = df_groups.aggregate(np.sum)
        df_totals['popn'] = 'All'
        df_LY = df_LY.append(df_totals, ignore_index=True, sort=False)

    df_LY = df_LY.drop(columns='draw_number')

    # Calculate relative gains.
    df_LY['LY_gain'] = df_LY['LY'] - df_LY['bau_LY']
    df_LY['LY_pcnt'] = 100 * df_LY['LY_gain'] / df_LY['bau_LY']
    df_LY['HALY_gain'] = df_LY['HALY'] - df_LY['bau_HALY']
    df_LY['HALY_pcnt'] = 100 * df_LY['HALY_gain'] / df_LY['bau_HALY']

    # NOTE: multiply ACMR by 1e5
    df_ACMR['acmr'] = 1e5 * df_ACMR['acmr']
    df_ACMR['bau_acmr'] = 1e5 * df_ACMR['bau_acmr']
    df_ACMR['acmr_gain'] = df_ACMR['acmr'] - df_ACMR['bau_acmr']
    df_ACMR['acmr_pcnt'] = 100 * df_ACMR['acmr_gain'] / df_ACMR['bau_acmr']

    df_YLDR['yld_rate_gain'] = df_YLDR['yld_rate'] - df_YLDR['bau_yld_rate']
    df_YLDR['yld_rate_pcnt'] = 100 * df_YLDR['yld_rate_gain'] / df_YLDR['bau_yld_rate']

    # Calculate CIs
    df_LY = calculate_ci(
        df_LY,
        ['bau_LY', 'LY_gain', 'LY_pcnt', 'bau_HALY', 'HALY_gain', 'HALY_pcnt'],
        ['popn', 'sex'])
    df_ACMR = calculate_ci(
        df_ACMR,
        ['bau_acmr', 'acmr_gain', 'acmr_pcnt'],
        ['popn', 'year', 'age', 'sex'])
    df_YLDR = calculate_ci(
        df_YLDR,
        ['bau_yld_rate', 'yld_rate_gain', 'yld_rate_pcnt'],
        ['popn', 'year', 'age', 'sex'])

    # Round numbers.
    suffixes = ['lower', 'upper', 'median', 'mean']
    prefix_LY = {
        'bau_LY': 0,
        'bau_HALY': 0,
        'LY_gain': 0,
        'HALY_gain': 0,
        'LY_pcnt': 2,
        'HALY_pcnt': 2,
    }
    prefix_ACMR = {
        'bau_acmr': 1,
        'acmr_gain': 1,
        'acmr_pcnt': 2,
    }
    prefix_YLDR = {
        'bau_yld_rate': 4,
        'yld_rate_gain': 4,
        'yld_rate_pcnt': 2,
    }
    round_LY = {'{}.{}'.format(prefix, suffix): digits
                for prefix, digits in prefix_LY.items()
                for suffix in suffixes}
    round_ACMR = {'{}.{}'.format(prefix, suffix): digits
                for prefix, digits in prefix_ACMR.items()
                for suffix in suffixes}
    round_YLDR = {'{}.{}'.format(prefix, suffix): digits
                for prefix, digits in prefix_YLDR.items()
                for suffix in suffixes}

    df_LY = df_LY.round(round_LY)
    df_ACMR = df_ACMR.round(round_ACMR)
    df_YLDR = df_YLDR.round(round_YLDR)

    # Add scenario metadata.
    for df in [df_LY, df_ACMR, df_YLDR]:
        df.insert(0, 'interv', interv)
        df.insert(0, 'tob_prev', tob_prev)
        df.insert(0, 'delay', delay)

    logging.info('Aggregated results from {} files'.format(len(file_list)))

    return {
        'LY': df_LY,
        'ACMR': df_ACMR,
        'YLDR': df_YLDR,
    }


def calculate_ci(df, ci_cols, group_cols, width=0.95):
    df_groups = df.groupby(group_cols)

    pr_lower = 0.5 * (1 - width)
    pr_upper = 0.5 * (1 + width)

    def lower(x):
        return x.quantile(pr_lower)

    def upper(x):
        return x.quantile(pr_upper)

    # Calculate the 95% CI, median, and mean values for each column.
    fns = [lower, upper, 'median', 'mean']
    df_ci = df_groups.agg(fns)

    # NOTE: aggregation yields multi-level column names, which we flatten.
    # So columns will be named 'LY_gain.upper', 'LY_gain.lower', etc.
    new_names = ['.'.join(col_levels) for col_levels in df_ci.columns.values]
    df_ci.columns = new_names

    # Restore the original grouping columns from the index.
    df_ci = df_ci.reset_index()

    return df_ci


def load_simulation_results():
    log = logging.getLogger(__name__)

    populations = ['maori', 'non-maori']
    interventions = ['erad', 'tfg', 'tax']
    bau_delay = [20, 0, 20]
    bau_tob_prev = ['decreasing', 'decreasing', 'constant']

    dfs_LY = []
    dfs_ACMR = []
    dfs_YLDR = []

    for (delay, tob_prev) in zip(bau_delay, bau_tob_prev):
        for interv in interventions:
            logging.info('Scenario is {} years, {} prev, {}'.format(
                delay, tob_prev, interv))
            dfs = summarise_results(delay, tob_prev, interv)
            if dfs:
                dfs_LY.append(dfs['LY'])
                dfs_ACMR.append(dfs['ACMR'])
                dfs_YLDR.append(dfs['YLDR'])

    return {
        'LY': pd.concat(dfs_LY, ignore_index=True, sort=False),
        'ACMR': pd.concat(dfs_ACMR, ignore_index=True, sort=False),
        'YLDR': pd.concat(dfs_YLDR, ignore_index=True, sort=False),
    }


def main(args=None):
    # Treat FutureWarning exceptions as errors.
    warnings.simplefilter('error')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    log = logging.getLogger(__name__)

    dfs = load_simulation_results()
    log.info('Loaded all simulation results')

    log.info('Sorting simulation results ...')
    df_LY = dfs['LY']
    df_ACMR= dfs['ACMR']
    df_YLDR = dfs['YLDR']

    sort_cols = ['delay', 'tob_prev', 'interv']
    df_LY = df_LY.sort_values(by=['sex', 'popn']).reset_index(drop=True)
    df_ACMR = df_ACMR.sort_values(by=sort_cols).reset_index(drop=True)
    df_YLDR = df_YLDR.sort_values(by=sort_cols).reset_index(drop=True)

    # Separate the LY and HALY data into two tables.
    ly_cols = [c for c in df_LY.columns if 'HALY' not in c]
    haly_cols = [c for c in df_LY.columns if 'HALY' in c or 'LY' not in c]
    df_HALY = df_LY.reindex(columns=haly_cols)
    df_LY = df_LY.reindex(columns=ly_cols)

    log.info('Saving results to disk')
    df_LY.to_csv('uncertainty-LY.csv', index=False)
    df_HALY.to_csv('uncertainty-HALY.csv', index=False)
    df_ACMR.to_csv('uncertainty-ACMR.csv', index=False)
    df_YLDR.to_csv('uncertainty-YLDR.csv', index=False)

    return 0


if __name__ == '__main__':
    sys.exit(main())
