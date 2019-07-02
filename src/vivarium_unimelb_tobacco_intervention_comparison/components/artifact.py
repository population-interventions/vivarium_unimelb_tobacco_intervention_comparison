import datetime
import logging
import numpy as np
import os
import os.path

from vivarium_public_health.dataset_manager import hdf
from vivarium_public_health.dataset_manager.artifact import Artifact

from .population import Population
from .disease import Diseases
from .risk_factor import Tobacco
from .uncertainty import Normal, Beta, LogNormal, LogNormalRawSD


def assemble_chd_only(num_draws):
    artifact_file = 'test_reduce_chd.hdf'

    year_start = 2011
    data_dir_maori = './data/maori'
    data_dir_nonmaori = './data/non-maori'
    prng = np.random.RandomState(seed=49430)

    def new_samples():
        return prng.random_sample(num_draws)

    dist_yld = LogNormal(sd_pcnt=10)
    smp_yld = new_samples()

    dist_chd_apc = Normal(sd_pcnt=0.5)
    smp_chd_apc = new_samples()
    dist_chd_i = Normal(sd_pcnt=5)
    smp_chd_i = new_samples()
    dist_chd_r = Normal(sd_pcnt=5)
    smp_chd_r = new_samples()
    dist_chd_f = Normal(sd_pcnt=5)
    smp_chd_f = new_samples()
    dist_chd_yld = Normal(sd_pcnt=10)
    smp_chd_yld = new_samples()
    dist_chd_prev = Normal(sd_pcnt=5)
    smp_chd_prev = new_samples()

    # Instantiate components for the non-Maori population.
    p_nm = Population(data_dir_nonmaori, year_start)
    l_nm = Diseases(data_dir_nonmaori, year_start, p_nm.year_end)

    # Create a new artifact file, replacing any existing file.
    if os.path.exists(artifact_file):
        os.remove(artifact_file)
    hdf.touch(artifact_file, append=False)
    art = Artifact(artifact_file)

    # 'population.structure'
    popn_nm = p_nm.get_population()
    art.write('population.structure',
              popn_nm)
    # 'cause.all_causes.disability_rate'
    popn_yld_nm = p_nm.sample_disability_rate_from(dist_yld, smp_yld)
    art.write('cause.all_causes.disability_rate',
              popn_yld_nm)
    # 'cause.all_causes.mortality'
    popn_acmr_nm = p_nm.get_mortality_rate()
    art.write('cause.all_causes.mortality',
              popn_acmr_nm)

    name = 'CHD'
    disease = l_nm.chronic[name]

    # 'chronic_disease.{}.incidence'
    chd_i = disease.sample_i_from(
        dist_chd_i, dist_chd_apc,
        smp_chd_i, smp_chd_apc)
    art.write('chronic_disease.CHD.incidence',
              check_for_bin_edges(chd_i))
    # 'chronic_disease.{}.remission'
    chd_r = disease.sample_r_from(
        dist_chd_r, dist_chd_apc,
        smp_chd_r, smp_chd_apc)
    art.write('chronic_disease.CHD.remission',
              check_for_bin_edges(chd_r))
    # 'chronic_disease.{}.mortality'
    chd_f = disease.sample_f_from(
        dist_chd_f, dist_chd_apc,
        smp_chd_f, smp_chd_apc)
    art.write('chronic_disease.CHD.mortality',
              check_for_bin_edges(chd_f))
    # 'chronic_disease.{}.morbidity'
    chd_yld = disease.sample_yld_from(
        dist_chd_yld, dist_chd_apc,
        smp_chd_yld, smp_chd_apc)
    art.write('chronic_disease.CHD.morbidity',
              check_for_bin_edges(chd_yld))
    # 'chronic_disease.{}.prevalence'
    chd_prev = disease.sample_prevalence_from(
        dist_chd_prev, smp_chd_prev)
    art.write('chronic_disease.CHD.prevalence',
              check_for_bin_edges(chd_prev))


def check_for_bin_edges(df):
    """
    Check that lower (inclusive) and upper (exclusive) bounds for year and age
    are defined as table columns.
    """

    if 'age_group_start' in df.columns and 'year_start' in df.columns:
        return df
    else:
        raise ValueError('Table does not have bins')


def assemble_tobacco_artifacts(num_draws, seed=49430):
    """
    Assemble the data artifacts required to simulate the various tobacco
    interventions.

    :param num_draws: The number of random draws to sample for each rate
        and quantity, for the uncertainty analysis.
    :param seed: The seed for the pseudo-random number generator used to
        generate the random samples.
    """
    year_start = 2011
    data_dir_maori = './data/maori'
    data_dir_nonmaori = './data/non-maori'
    prng = np.random.RandomState(seed=seed)
    logger = logging.getLogger(__name__)

    # Instantiate components for the non-Maori population.
    p_nm = Population(data_dir_nonmaori, year_start)
    l_nm = Diseases(data_dir_nonmaori, year_start, p_nm.year_end)
    t_nm = Tobacco(data_dir_nonmaori, year_start, p_nm.year_end)

    # Instantiate components for the Maori population.
    p_m = Population(data_dir_maori, year_start)
    l_m = Diseases(data_dir_maori, year_start, p_m.year_end)
    t_m = Tobacco(data_dir_maori, year_start, p_m.year_end)

    # Define data structures to record the samples from the unit interval that
    # are used to sample each rate/quantity, so that they can be correlated
    # across both populations.
    smp_yld = prng.random_sample(num_draws)
    smp_chronic_apc = {}
    smp_chronic_i = {}
    smp_chronic_r = {}
    smp_chronic_f = {}
    smp_chronic_yld = {}
    smp_chronic_prev = {}
    smp_acute_f = {}
    smp_acute_yld = {}
    smp_tob_dis_tbl = {}
    smp_tob_i = prng.random_sample(num_draws)
    smp_tob_r = prng.random_sample(num_draws)
    smp_tob_elast = prng.random_sample(num_draws)
    smp_tob_elast_maori = prng.random_sample(num_draws)

    # Define the sampling distributions in terms of their family and their
    # *relative* standard deviation; they will be used to draw samples for
    # both populations.
    dist_yld = LogNormal(sd_pcnt=10)
    dist_chronic_apc = Normal(sd_pcnt=0.5)
    dist_chronic_i = Normal(sd_pcnt=5)
    dist_chronic_r = Normal(sd_pcnt=5)
    dist_chronic_f = Normal(sd_pcnt=5)
    dist_chronic_yld = Normal(sd_pcnt=10)
    dist_chronic_prev = Normal(sd_pcnt=5)
    dist_acute_f = Normal(sd_pcnt=10)
    dist_acute_yld = Normal(sd_pcnt=10)
    dist_tob_i = Beta(sd_pcnt=20)
    dist_tob_r = Beta(sd_pcnt=20)
    dist_tob_elast = Normal(sd_pcnt=20)
    dist_tob_elast_maori_scale = Normal(sd_pcnt=10)
    tob_elast_maori_scale = 1.2

    logger.info('{} Generating samples'.format(
        datetime.datetime.now().strftime("%H:%M:%S")))

    for name, disease_nm in l_nm.chronic.items():
        # Draw samples for each rate/quantity for this disease.
        smp_chronic_apc[name] = prng.random_sample(num_draws)
        smp_chronic_i[name] = prng.random_sample(num_draws)
        smp_chronic_r[name] = prng.random_sample(num_draws)
        smp_chronic_f[name] = prng.random_sample(num_draws)
        smp_chronic_yld[name] = prng.random_sample(num_draws)
        smp_chronic_prev[name] = prng.random_sample(num_draws)

        # Also draw samples for the RR associated with tobacco smoking.
        smp_tob_dis_tbl[name] = prng.random_sample(num_draws)

    for name, disease_nm in l_nm.acute.items():
        # Draw samples for each rate/quantity for this disease.
        smp_acute_f[name] = prng.random_sample(num_draws)
        smp_acute_yld[name] = prng.random_sample(num_draws)

        # Also draw samples for the RR associated with tobacco smoking.
        smp_tob_dis_tbl[name] = prng.random_sample(num_draws)

    # Now write all of the required tables for:
    #
    #   - Both the Maori and non-Maori populations; and
    #   - Both the 20-year and 0-year recovery from smoking.
    #
    # So there are 4 data artifacts to write.

    nm_artifact_fmt = 'mslt_tobacco_non-maori_{}-years.hdf'
    m_artifact_fmt = 'mslt_tobacco_maori_{}-years.hdf'

    logger.info('{} Generating artifacts'.format(
        datetime.datetime.now().strftime("%H:%M:%S")))

    for recovery in [20, 0]:
        nm_artifact_file = nm_artifact_fmt.format(recovery)
        m_artifact_file = m_artifact_fmt.format(recovery)

        exposure = 'tobacco'

        # Initialise each artifact file.
        for file_name in [nm_artifact_file, m_artifact_file]:
            if os.path.exists(file_name):
                os.remove(file_name)
            hdf.touch(file_name, append=False)

        # Write the data tables to each artifact file.
        art_nm = Artifact(nm_artifact_file)
        art_m = Artifact(m_artifact_file)

        logger.info('{} Writing population tables'.format(
            datetime.datetime.now().strftime("%H:%M:%S")))

        # Write the main population tables.
        write_table(art_nm, 'population.structure',
                     p_nm.get_population())
        write_table(art_m, 'population.structure',
                    p_m.get_population())
        write_table(art_nm, 'cause.all_causes.disability_rate',
                     p_nm.sample_disability_rate_from(dist_yld, smp_yld))
        write_table(art_m, 'cause.all_causes.disability_rate',
                    p_m.sample_disability_rate_from(dist_yld, smp_yld))
        write_table(art_nm, 'cause.all_causes.mortality',
                     p_nm.get_mortality_rate())
        write_table(art_m, 'cause.all_causes.mortality',
                    p_m.get_mortality_rate())

        # Write the chronic disease tables.
        for name, disease_nm in l_nm.chronic.items():
            logger.info('{} Writing tables for {}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), name))

            disease_m = l_m.chronic[name]

            write_table(art_nm, 'chronic_disease.{}.incidence'.format(name),
                         disease_nm.sample_i_from(
                             dist_chronic_i, dist_chronic_apc,
                             smp_chronic_i[name], smp_chronic_apc[name]))
            write_table(art_m, 'chronic_disease.{}.incidence'.format(name),
                        disease_m.sample_i_from(
                            dist_chronic_i, dist_chronic_apc,
                            smp_chronic_i[name], smp_chronic_apc[name]))
            write_table(art_nm, 'chronic_disease.{}.remission'.format(name),
                         disease_nm.sample_r_from(
                             dist_chronic_r, dist_chronic_apc,
                             smp_chronic_r[name], smp_chronic_apc[name]))
            write_table(art_m, 'chronic_disease.{}.remission'.format(name),
                         disease_m.sample_r_from(
                             dist_chronic_r, dist_chronic_apc,
                             smp_chronic_r[name], smp_chronic_apc[name]))
            write_table(art_nm, 'chronic_disease.{}.mortality'.format(name),
                         disease_nm.sample_f_from(
                             dist_chronic_f, dist_chronic_apc,
                             smp_chronic_f[name], smp_chronic_apc[name]))
            write_table(art_m, 'chronic_disease.{}.mortality'.format(name),
                        disease_m.sample_f_from(
                            dist_chronic_f, dist_chronic_apc,
                            smp_chronic_f[name], smp_chronic_apc[name]))
            write_table(art_nm, 'chronic_disease.{}.morbidity'.format(name),
                         disease_nm.sample_yld_from(
                             dist_chronic_yld, dist_chronic_apc,
                             smp_chronic_yld[name], smp_chronic_apc[name]))
            write_table(art_m, 'chronic_disease.{}.morbidity'.format(name),
                        disease_m.sample_yld_from(
                            dist_chronic_yld, dist_chronic_apc,
                            smp_chronic_yld[name], smp_chronic_apc[name]))
            write_table(art_nm, 'chronic_disease.{}.prevalence'.format(name),
                         disease_nm.sample_prevalence_from(
                             dist_chronic_prev, smp_chronic_prev[name]))
            write_table(art_m, 'chronic_disease.{}.prevalence'.format(name),
                         disease_m.sample_prevalence_from(
                             dist_chronic_prev, smp_chronic_prev[name]))

        # Write the acute disease tables.
        for name, disease_nm in l_nm.acute.items():
            logger.info('{} Writing tables for {}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), name))

            disease_m = l_m.acute[name]

            write_table(art_nm, 'acute_disease.{}.mortality'.format(name),
                         disease_nm.sample_excess_mortality_from(
                             dist_acute_f, smp_acute_f[name]))
            write_table(art_m, 'acute_disease.{}.mortality'.format(name),
                         disease_m.sample_excess_mortality_from(
                             dist_acute_f, smp_acute_f[name]))
            write_table(art_nm, 'acute_disease.{}.morbidity'.format(name),
                         disease_nm.sample_disability_from(
                             dist_acute_yld, smp_acute_yld[name]))
            write_table(art_m, 'acute_disease.{}.morbidity'.format(name),
                         disease_m.sample_disability_from(
                             dist_acute_yld, smp_acute_yld[name]))

        # Write the risk factor tables.
        for name in [exposure]:
            logger.info('{} Writing tables for {}'.format(
                datetime.datetime.now().strftime("%H:%M:%S"), name))

            write_table(art_nm, 'risk_factor.{}.incidence'.format(name),
                         t_nm.sample_i_from(dist_tob_i, smp_tob_i))
            write_table(art_m, 'risk_factor.{}.incidence'.format(name),
                        t_m.sample_i_from(dist_tob_i, smp_tob_i))
            write_table(art_nm, 'risk_factor.{}.remission'.format(name),
                         t_nm.sample_r_from(dist_tob_r, smp_tob_r))
            write_table(art_m, 'risk_factor.{}.remission'.format(name),
                        t_m.sample_r_from(dist_tob_r, smp_tob_r))

            if recovery == 0:
                # Cessation confers immediate recovery.
                write_table(art_nm, 'risk_factor.{}.mortality_relative_risk'.format(name),
                             collapse_tobacco_mortality_rr(
                                 t_nm.get_expected_mortality_rr(),
                                 name))
                write_table(art_m, 'risk_factor.{}.mortality_relative_risk'.format(name),
                            collapse_tobacco_mortality_rr(
                                t_m.get_expected_mortality_rr(),
                                name))
                write_table(art_nm, 'risk_factor.{}.disease_relative_risk'.format(name),
                             collapse_tobacco_disease_rr(
                                 t_nm.sample_disease_rr_from(smp_tob_dis_tbl)))
                write_table(art_m, 'risk_factor.{}.disease_relative_risk'.format(name),
                            collapse_tobacco_disease_rr(
                                t_m.sample_disease_rr_from(smp_tob_dis_tbl)))
                write_table(art_nm, 'risk_factor.{}.prevalence'.format(name),
                             collapse_tobacco_prevalence(
                                 t_nm.get_expected_prevalence(),
                                 name))
                write_table(art_m, 'risk_factor.{}.prevalence'.format(name),
                             collapse_tobacco_prevalence(
                                 t_m.get_expected_prevalence(),
                                 name))
            else:
                write_table(art_nm, 'risk_factor.{}.mortality_relative_risk'.format(name),
                             t_nm.get_expected_mortality_rr())
                write_table(art_m, 'risk_factor.{}.mortality_relative_risk'.format(name),
                            t_m.get_expected_mortality_rr())
                write_table(art_nm, 'risk_factor.{}.disease_relative_risk'.format(name),
                             t_nm.sample_disease_rr_from(smp_tob_dis_tbl))
                write_table(art_m, 'risk_factor.{}.disease_relative_risk'.format(name),
                            t_m.sample_disease_rr_from(smp_tob_dis_tbl))
                write_table(art_nm, 'risk_factor.{}.prevalence'.format(name),
                             t_nm.get_expected_prevalence())
                write_table(art_m, 'risk_factor.{}.prevalence'.format(name),
                            t_m.get_expected_prevalence())

            logger.info('{}     Tax effects (non-Maori)'.format(
                datetime.datetime.now().strftime("%H:%M:%S")))
            tob_elast_nm = t_nm.sample_price_elasticity_from(
                dist_tob_elast, smp_tob_elast)

            tob_tax_nm = t_nm.sample_tax_effects_from_elasticity_wide(tob_elast_nm)
            incidence_cols = [c for c in tob_tax_nm.columns
                              if c != 'remission_effect']
            remission_cols = [c for c in tob_tax_nm.columns
                              if c != 'incidence_effect']
            write_table(art_nm, 'risk_factor.{}.tax_effect_incidence'.format(name),
                         tob_tax_nm.loc[:, incidence_cols])
            write_table(art_nm, 'risk_factor.{}.tax_effect_remission'.format(name),
                         tob_tax_nm.loc[:, remission_cols])
            del tob_tax_nm

            logger.info('{}     Tax effects (Maori)'.format(
                datetime.datetime.now().strftime("%H:%M:%S")))
            tob_elast_m = t_m.scale_price_elasticity_from(
                tob_elast_nm,
                tob_elast_maori_scale,
                Normal(sd_pcnt=10),
                smp_tob_elast_maori)

            tob_tax_m = t_m.sample_tax_effects_from_elasticity_wide(tob_elast_m)
            incidence_cols = [c for c in tob_tax_m.columns
                              if c != 'remission_effect']
            remission_cols = [c for c in tob_tax_m.columns
                              if c != 'incidence_effect']
            write_table(art_m, 'risk_factor.{}.tax_effect_incidence'.format(name),
                        tob_tax_m.loc[:, incidence_cols])
            write_table(art_m, 'risk_factor.{}.tax_effect_remission'.format(name),
                        tob_tax_m.loc[:, remission_cols])
            del tob_tax_m
            del tob_elast_nm
            del tob_elast_m

        print(nm_artifact_file)
        print(m_artifact_file)


def write_table(artifact, path, data):
    """
    Write a data table to an artifact, after ensuring that it doesn't contain
    any NA values.

    :param artifact: The artifact object.
    :param path: The table path.
    :param data: The table data.
    """
    if np.any(data.isna()):
        msg = 'NA values in table {} for {}'.format(path, artifact.path)
        raise ValueError(msg)

    logger = logging.getLogger(__name__)
    logger.info('{} Writing table {} to {}'.format(
        datetime.datetime.now().strftime("%H:%M:%S"), path, artifact.path))
    artifact.write(path, data)


def collapse_tobacco_prevalence(data, exposure):
    """
    Collapse the tunnel states into a single post-cessation state.

    :param data: The prevalence data.
    :param exposure: The name of the exposure.
    """
    all_columns = data.columns

    prefix = '{}.'.format(exposure)
    suffixes = ['no', 'yes', '0']

    prev_cols = [prefix + suffix for suffix in suffixes]
    idx_cols = [c for c in all_columns if not c.startswith(exposure)]

    want_cols = idx_cols + prev_cols
    keep_cols = [c for c in all_columns if c in want_cols]
    data = data.loc[:, keep_cols]

    # Ensure that each row sums to unity.
    final_col = prefix + '0'
    other_prev_cols = [c for c in prev_cols if c != final_col]
    data.loc[:, final_col] = 1.0 - data.loc[:, other_prev_cols].sum(axis=1)

    return data


def collapse_tobacco_mortality_rr(data, exposure):
    """
    Make remission instantly confer a relative risk of 1.0.

    :param data: The mortality relative risk data.
    :param exposure: The name of the exposure.
    """
    all_columns = data.columns

    bau_prefix = '{}.'.format(exposure)
    int_prefix = '{}_intervention.'.format(exposure)
    suffixes = ['no', 'yes', '0']

    bau_cols = [bau_prefix + suffix for suffix in suffixes]
    int_cols = [int_prefix + suffix for suffix in suffixes]
    idx_cols = [c for c in all_columns if not c.startswith(exposure)]

    want_cols = idx_cols + bau_cols + int_cols
    keep_cols = [c for c in all_columns if c in want_cols]
    data = data.loc[:, keep_cols]

    bau_final = bau_prefix + '0'
    int_final = int_prefix + '0'
    data.loc[:, bau_final] = 1.0
    data.loc[:, int_final] = 1.0

    return data


def collapse_tobacco_disease_rr(data):
    """
    Make remission instantly confer a relative risk of 1.0.

    :param data: The disease incidence relative risk data.
    """
    all_columns = data.columns

    diseases = list({c[:-4] for c in all_columns if c.endswith('_yes')})
    suffixes = ['no', 'yes', 'post_0']

    for disease in diseases:
        prefix = '{}_'.format(disease)
        dis_cols = [prefix + suffix for suffix in suffixes]
        drop_cols = [c for c in all_columns
                     if c.startswith(prefix) and c not in dis_cols]
        data = data.drop(columns=drop_cols)
        final_col = prefix + 'post_0'
        data.loc[:, final_col] = 1.0

    return data
