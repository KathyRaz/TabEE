import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from constants import NUMERIC_TYPE, CATEGORICAL_TYPE, LIFT_SUFFIX


def calculate_distance(s, r, data_type):
    s = np.array(s)
    s = s[s == s]
    r = np.array(r)
    r = r[r == r]
    # if data_type == NUMERIC_TYPE:
    #     return 0 if len(r) == 0 else wasserstein_distance(r, s)
    # else:
    return 0 if len(r) == 0 else jensenshannon(r, s)


def normalize_by_clusters_num(raw_score, num_clusters):
    return (raw_score - 1.0 / num_clusters) / (1.0 - 1.0 / num_clusters)


def determine_column_type(column_series):
    if (column_series.dtype == 'string' or column_series.dtype == 'object') or (
            len(column_series.drop_duplicates()) < 20):
        return CATEGORICAL_TYPE
    return NUMERIC_TYPE


def equalObs(x, nbin):
    nlen = len(x)
    interp = np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))
    return np.unique(interp)


def distributions_distance(source_dist, result_dist, data_type):
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    dist = calculate_distance(distributions_pd[source_dist.name], distributions_pd[result_dist.name], data_type)
    return dist, distributions_pd[result_dist.name].fillna(0.0)


def distributions_lift(source_dist, result_dist):
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    return distributions_pd[result_dist.name] / distributions_pd[source_dist.name]
