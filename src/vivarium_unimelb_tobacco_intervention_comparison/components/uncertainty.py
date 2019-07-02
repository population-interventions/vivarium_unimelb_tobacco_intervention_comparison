"""Add support for uncertainty analyses by varying input data."""

import pandas as pd
import numpy as np
import re
import scipy.stats as st


__draw_re = re.compile('.+_draw_[0-9]+$')
"""
A regular expression that matches column names which contain a draw for some
quantity.
"""


__draw_num_re = re.compile('_[0-9]+$')
"""
A regular expression that matches the draw number suffix of these columns.
"""


__suffix_re = re.compile('_draw$')
"""
A regular expression that matches the suffix of the column stubs.
"""


def sample_column_long(data, column, dist, samples):
    if len(samples.shape) != 1:
        raise ValueError('Samples must be a one-dimensional array')
    if np.any(samples < 0.0) or np.any(samples >= 1.0):
        raise ValueError('Samples lie outside of [0, 1)')

    index_cols = ['age', 'sex']
    df = data.loc[:, index_cols].copy()

    if np.any(df.isna()):
        raise ValueError('NA values found, sampling column {}'.format(column))

    # The number of draws, *also* including draw 0 (the mean value).
    samples = np.insert(samples, 0, 0.5)
    num_draws = len(samples)
    df = df.reindex(df.index.repeat(num_draws)).reset_index(drop=True)

    mean_values = data[column]
    sample_shape = len(mean_values) * num_draws

    all_zeros = len(mean_values.nonzero()[0]) == 0
    if all_zeros:
        sampled_values = np.zeros(sample_shape)
    else:
        # An array of shape ``(n, v)`` for ``n`` percentile samples
        # and ``v`` mean values.
        sampled_values = dist.correlated_samples(mean_values, samples)
        sampled_values[:, mean_values == 0.0] = 0
        # Ensure that draw #0 is the expected value.
        sampled_values[0, :] = mean_values
        sampled_values = sampled_values.reshape(sample_shape, order='F')

    df['draw'] = np.tile(range(num_draws), len(mean_values))
    df[column] = sampled_values

    df = df.sort_values(['age', 'sex', 'draw'])
    df = df.reset_index(drop=True)

    if np.any(df.isna()):
        raise ValueError('NA values found, sampling rate {}'.format(column))

    return df


def sample_fixed_rate_from(year_start, year_end, data, rate_name,
                           rate_dist, samples):
    """
    Draw correlated samples for a rate at each year.

    :param year_start: The year at which the simulation starts.
    :param year_end: The year at which the simulation ends.
    :param data: The data table that contains the rate values.
    :param rate_name: The column name that defines the mean values.
    :param rate_dist: The uncertainty distribution for the rate values.
    :param samples: Random samples drawn from the half-open interval [0, 1).
    """
    value_col = rate_name

    # Sample the initial rate for each cohort.
    df = sample_column_long(data, value_col, rate_dist, samples)

    df.insert(0, 'year_start', year_start)
    df.insert(1, 'year_end', year_end + 1)
    df = df.rename(columns={'age': 'age_group_start'})
    df.insert(df.columns.get_loc('age_group_start') + 1,
              'age_group_end',
              df['age_group_start'] + 1)

    df = df.sort_values(['year_start', 'age_group_start', 'sex', 'draw'])
    df = df.reset_index(drop=True)

    if np.any(df.isna()):
        raise ValueError('NA values found, sampling rate {}'.format(column))

    return df


class Normal:
    """
    Draw samples from a normal distribution, whose standard deviation is a
    percentage of the mean.

    :param sd_pcnt: The standard deviation, represented as a percentage of the
        mean; valid values are between ``0.1`` and ``20``.
    """

    _sd_min = 0.1
    _sd_max = 20

    def __init__(self, sd_pcnt):
        if sd_pcnt < self._sd_min or sd_pcnt > self._sd_max:
            raise ValueError('SD% of {} is not between {}% and {}%'.format(
                sd_pcnt, self._sd_min, self._sd_max))
        self.sd_frac = sd_pcnt / 100.0

    def uncorrelated_samples(self, mean_values, num_samples, prng):
        """
        Draw **uncorrelated** values at random.

        :param mean_values: The mean values.
        :param num_samples: The number of samples to return.
        :param prng: A ``numpy.random.RandomState`` instance.
        :returns: An array of shape ``(n, v)`` for ``n`` samples and ``v``
            mean values.
        """
        mean = mean_values.values
        sd = np.absolute(mean) * self.sd_frac
        size = (num_samples, ) + mean.shape
        return prng.normal(mean[np.newaxis, ...],
                           sd[np.newaxis, ...],
                           size=size)

    def correlated_samples(self, mean_values, samples):
        """
        Draw **correlated** values according to the provided percentiles.

        :param mean_values: The mean values.
        :param samples: The percentiles.
        :returns: An array of shape ``(n, v)`` for ``n`` percentile samples
            and ``v`` mean values.
        """
        mean = mean_values.values
        zero_mask = mean == 0.0
        sd = np.absolute(mean) * self.sd_frac
        if np.any(zero_mask):
            # Only draw samples where the SD is non-zero.
            values = np.zeros(samples.shape + mean.shape)
            nonz_mask = ~ zero_mask
            rv = st.norm(loc=mean[nonz_mask], scale=sd[nonz_mask])
            values[:, nonz_mask] = rv.ppf(samples[..., np.newaxis])
            return values
        else:
            rv = st.norm(loc=mean, scale=sd)
            return rv.ppf(samples[..., np.newaxis])


class LogNormal:
    """
    Draw samples from a log-normal distribution, whose standard deviation is a
    percentage of the mean.

    :param sd_pcnt: The standard deviation, represented as a percentage of the
        mean *of the underlying normal distribution*; valid values are between
        ``0.1`` and ``20``.
    """

    _sd_min = 0.1
    _sd_max = 20

    def __init__(self, sd_pcnt):
        if sd_pcnt < self._sd_min or sd_pcnt > self._sd_max:
            raise ValueError('SD% of {} is not between {}% and {}%'.format(
                sd_pcnt, self._sd_min, self._sd_max))
        self.sd_frac = sd_pcnt / 100.0

    def uncorrelated_samples(self, mean_values, num_samples, prng):
        """
        Draw **uncorrelated** values at random.

        :param mean_values: The mean values.
        :param num_samples: The number of samples to return.
        :param prng: A ``numpy.random.RandomState`` instance.
        :returns: An array of shape ``(n, v)`` for ``n`` samples and ``v``
            mean values.
        """
        log_mean = np.log(mean_values.values)
        log_sd = np.absolute(log_mean) * self.sd_frac
        size = (num_samples, ) + log_mean.shape
        # NOTE: RandomState.lognormal takes the mean and SD of the underlying
        # normal distribution.
        return prng.lognormal(mean=log_mean[np.newaxis, ...],
                              sigma=log_sd[np.newaxis, ...],
                              size=size)

    def correlated_samples(self, mean_values, samples):
        """
        Draw **correlated** values according to the provided percentiles.

        :param mean_values: The mean values.
        :param samples: The percentiles.
        :returns: An array of shape ``(n, v)`` for ``n`` percentile samples
            and ``v`` mean values.
        """
        log_mean = np.log(mean_values.values)
        log_sd = np.absolute(log_mean) * self.sd_frac
        zero_mask = log_mean == 0.0
        # NOTE: scipy.stats.lognorm takes the *exponential* mean, and the SD
        # of the underlying normal distribution.
        if np.any(zero_mask):
            # Only draw samples where the SD is non-zero.
            values = np.ones(samples.shape + log_mean.shape)
            nonz_mask = ~ zero_mask
            rv = st.lognorm(scale=mean_values.values[nonz_mask],
                            s=log_sd[nonz_mask])
            values[:, nonz_mask] = rv.ppf(samples[..., np.newaxis])
            return values
        else:
            rv = st.lognorm(scale=mean_values.values, s=log_sd)
            return rv.ppf(samples[..., np.newaxis])


class LogNormalRawSD:
    """
    Draw samples from a log-normal distribution, where the standard deviation
    is defined separately for each mean value.

    :param log_sd: The standard deviation values.
    """

    def __init__(self, log_sd):
        self._log_sd = log_sd

    def uncorrelated_samples(self, mean_values, num_samples, prng):
        """
        Draw **uncorrelated** values at random.

        :param mean_values: The mean values.
        :param num_samples: The number of samples to return.
        :param prng: A ``numpy.random.RandomState`` instance.
        :returns: An array of shape ``(n, v)`` for ``n`` samples and ``v``
            mean values.
        """
        log_sd = self._log_sd
        if mean_values.shape != log_sd.shape:
            raise ValueError('Mean and SD have different shapes')
        log_mean = np.log(mean_values.values)
        size = (num_samples, ) + log_mean.shape
        # NOTE: RandomState.lognormal takes the mean and SD of the underlying
        # normal distribution.
        return prng.lognormal(mean=log_mean[np.newaxis, ...],
                              sigma=log_sd[np.newaxis, ...],
                              size=size)

    def correlated_samples(self, mean_values, samples):
        """
        Draw **correlated** values according to the provided percentiles.

        :param mean_values: The mean values.
        :param samples: The percentiles.
        :returns: An array of shape ``(n, v)`` for ``n`` percentile samples
            and ``v`` mean values.
        """
        log_sd = self._log_sd
        if mean_values.shape != log_sd.shape:
            raise ValueError('Mean and SD have different shapes')
        log_mean = np.log(mean_values.values)
        zero_mask = log_mean == 0.0
        # NOTE: scipy.stats.lognorm takes the *exponential* mean, and the SD
        # of the underlying normal distribution.
        if np.any(zero_mask):
            # Only draw samples where the SD is non-zero.
            values = np.ones(samples.shape + log_mean.shape)
            nonz_mask = ~ zero_mask
            rv = st.lognorm(scale=mean_values.values[nonz_mask],
                            s=log_sd[nonz_mask])
            values[:, nonz_mask] = rv.ppf(samples[..., np.newaxis])
            return values
        else:
            rv = st.lognorm(scale=mean_values.values, s=log_sd)
            return rv.ppf(samples[..., np.newaxis])


class Beta:
    """
    Draw samples from a Beta distribution, whose standard deviation is a
    percentage of the mean.

    :param sd_pcnt: The standard deviation, represented as a percentage of the
        mean; valid values are between ``0.1`` and ``20``.
    """

    _sd_min = 0.1
    _sd_max = 20

    def __init__(self, sd_pcnt):
        if sd_pcnt < self._sd_min or sd_pcnt > self._sd_max:
            raise ValueError('SD% of {} is not between {}% and {}%'.format(
                sd_pcnt, self._sd_min, self._sd_max))
        self.sd_frac = sd_pcnt / 100.0

    def shape_parameters(self, mean_values):
        """Calculate the shape parameters alpha and beta."""
        mean = mean_values.values
        sd = np.absolute(mean) * self.sd_frac
        varn = np.square(sd)
        nu = mean * (1 - mean) / varn - 1
        alpha = mean * nu
        beta = (1 - mean) * nu
        return (alpha, beta)


    def uncorrelated_samples(self, mean_values, num_samples, prng):
        """
        Draw **uncorrelated** values at random.

        :param mean_values: The mean values.
        :param num_samples: The number of samples to return.
        :param prng: A ``numpy.random.RandomState`` instance.
        :returns: An array of shape ``(n, v)`` for ``n`` samples and ``v``
            mean values.
        """
        size = (num_samples, ) + mean_values.shape
        alpha, beta = self.shape_parameters(mean_values)
        return prng.beta(a=alpha[np.newaxis, ...],
                         b=beta[np.newaxis, ...],
                         size=size)

    def correlated_samples(self, mean_values, samples):
        """
        Draw **correlated** values according to the provided percentiles.

        :param mean_values: The mean values.
        :param samples: The percentiles.
        :returns: An array of shape ``(n, v)`` for ``n`` percentile samples
            and ``v`` mean values.
        """
        zero_mask = mean_values.values == 0.0
        if np.any(zero_mask):
            # Only draw samples where the SD is non-zero.
            values = np.zeros(samples.shape + mean_values.shape)
            nonz_mask = ~ zero_mask
            alpha, beta = self.shape_parameters(mean_values[nonz_mask])
            rv = st.beta(a=alpha, b=beta)
            values[:, nonz_mask] = rv.ppf(samples[..., np.newaxis])
            return values
        else:
            alpha, beta = self.shape_parameters(mean_values)
            rv = st.beta(a=alpha, b=beta)
            return rv.ppf(samples[..., np.newaxis])


def test(num=10):
    """Select samples using both methods."""
    prng = np.random.RandomState(seed=49430)

    sd_pcnt = 5
    n = Normal(sd_pcnt)
    print('Normal')

    x = pd.Series(range(1, 6))
    print('x = {}'.format(x.values))
    y = n.uncorrelated_samples(x, num, prng)
    print(y)
    print()

    samples = prng.random_sample(size=num)
    print('percentiles = {}'.format(samples))
    y = n.correlated_samples(x, samples)
    print(y)
    print()

    b = Beta(sd_pcnt)
    print('Beta')

    x = pd.Series([0.1, 0.3, 0.5])
    print('x = {}'.format(x.values))
    y = b.uncorrelated_samples(x, num, prng)
    print(y)
    print()

    samples = prng.random_sample(size=num)
    print('percentiles = {}'.format(samples))
    y = b.correlated_samples(x, samples)
    print(y)
    print()

    sd_pcnt = 10
    l = LogNormal(sd_pcnt)
    print('LogNormal')

    x = pd.Series([1.0, 1.1, 1.5, 5.0])
    print('x = {}'.format(x.values))
    y = l.uncorrelated_samples(x, num, prng)
    print(y)
    print()

    samples = prng.random_sample(size=num)
    print('percentiles = {}'.format(samples))
    y = l.correlated_samples(x, samples)
    print(y)
    print()

    # Construct a table by adding each draw column in turn.
    df = pd.DataFrame(x, columns=['x'])
    for ix, col_values in enumerate(y):
        col_name = 'draw_{}'.format(ix)
        df[col_name] = col_values
    print(df.to_string())
    print()

    # Construct a table by concatenating all draw columns in one action.
    df2 = pd.DataFrame(x, columns=['x'])
    df_draws = pd.DataFrame(np.transpose(y))
    df_draws.columns = ['draw_{}'.format(ix) for ix in range(len(y))]
    df2 = pd.concat([df2, df_draws], axis=1)
    print(df2.to_string())
    print()
