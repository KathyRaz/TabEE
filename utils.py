import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from constants import NUMERIC_TYPE, CATEGORICAL_TYPE, LIFT_SUFFIX


def calculate_distance(s, r, data_type=None):
    s_np = np.array(s, dtype='float64')
    s_np = s_np[s_np == s_np]
    r_np = np.array(r, dtype='float64')
    r_np = r_np[r_np == r_np]
    # if data_type == NUMERIC_TYPE:
    #     return 0 if len(r) == 0 else wasserstein_distance(r, s)
    # else:
    try:
        return 0 if len(r_np) == 0 else jensenshannon(r_np, s_np)
    except Exception:
        return


def distributions_distance(source_dist, result_dist, data_type=None):
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    dist = calculate_distance(distributions_pd[source_dist.name], distributions_pd[result_dist.name], data_type)
    return dist, distributions_pd[result_dist.name].fillna(0.0)


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


def distributions_lift(source_dist, result_dist):
    # print(f"source:\\n {source_dist}")
    # print(f"result:\\n {result_dist}")
    distributions_pd = pd.concat([source_dist, result_dist], axis=1).fillna(0.0)
    return distributions_pd[result_dist.name] / distributions_pd[source_dist.name]


def diversity_from_distributions(distributions_list):
    diversity = 1
    for i in range(1, len(distributions_list), 1):
        min_distance = np.inf
        for j in range(i):
            distance = calculate_distance(distributions_list[i], distributions_list[j])
            min_distance = distance if distance < min_distance else min_distance
        diversity += min_distance
    return diversity


def diversity_metric(col_names, col_distributions):
    np_names = np.array(col_names)
    np_distributions = np.array(col_distributions, dtype='object')
    unique_names = np.unique(np_names)
    diversity = 0
    for name in unique_names:
        distributions = np_distributions[np_names == name]
        if len(distributions) == 1:
            diversity += 1
        else:  # TODO add permutation maximization for symmetry + scaling
            diversity += diversity_from_distributions(distributions)
    return diversity


def bin_single_column(binning_column, type, data_type, num_bins=10, bins=None):
    # data_type = self.source_distributions[binning_column.name][DATA_TYPE]
    # type must be one of 'source', 'result'
    if data_type == CATEGORICAL_TYPE:
        histogram = binning_column.value_counts(normalize=True).rename(f"{binning_column.name}_{type}")
        bins = list(histogram.index)
        return histogram, bins
    elif data_type == NUMERIC_TYPE:
        if bins is None and num_bins is not None:
            bins = equalObs(binning_column, num_bins)
        counts, bin_edges = np.histogram(binning_column, bins=bins)
        counts = counts / sum(counts)
        bin_edges[len(bin_edges) - 1] = bin_edges[len(bin_edges) - 1] + 1  # patch to solve pd.cut problem last bin
        return pd.Series(counts, index=bin_edges[:-1], name=f"{binning_column.name}_{type}"), bins
    else:
        raise Exception("incorrect data type")


def best_from_list_by_order_type(lst, ordering_type):
    assert ordering_type in ['best', 'worst', 'median']
    if ordering_type == 'best':
        return np.argmax(lst)
    elif ordering_type == 'median':
        return np.argsort(lst)[len(lst) // 2]
    elif ordering_type == 'worst':
        return np.argsort(lst)[0]
