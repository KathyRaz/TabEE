import numpy as np
import pandas as pd
from kstest import ks_2samp
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import pairwise_distances, calinski_harabasz_score, silhouette_score
from scipy.spatial.distance import jensenshannon as js_div
from pathos.pools import ParallelPool
from scipy.stats import wasserstein_distance
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
# Path hack.
import sys, os

sys.path.append(os.path.abspath('..') + "\\FEDEx\\")
from UserStudyInteractive import filter_


def unzip_bins(zipped_vals):
    bins = []
    for _, current_bin, _ in zipped_vals:
        bins.append(current_bin)
    return bins


def assign_probabilites(clustered_df, bins):
    explain_col = bins.get_binned_result_column().name
    binned_explain_col = explain_col + '_BINNED'
    proba_vals = bins.get_binned_result_column().value_counts() / bins.get_binned_result_column().value_counts().sum()
    value_bins = list(np.unique(bins.get_binned_result_column().values))

    assigned_clustered_df = clustered_df.copy()
    # binning
    assigned_clustered_df[binned_explain_col] = assigned_clustered_df[explain_col]
    if type(value_bins[0]) != str:
        for val_bin in value_bins:
            assigned_clustered_df[binned_explain_col] = assigned_clustered_df.apply(
                lambda x: val_bin if x[explain_col] > val_bin else x[binned_explain_col], axis=1)

    # assigning each bin its probability
    binned_proba = pd.DataFrame(proba_vals).reset_index().rename(
        {'index': binned_explain_col, explain_col: 'VAL_PROBA'}, axis=1)

    # assigning each row its probability
    assigned_clustered_df = assigned_clustered_df.merge(binned_proba, on=binned_explain_col, how='left').fillna(0.0)

    # assigning the A function
    assigned_clustered_df['A_VAL'] = assigned_clustered_df['VAL_PROBA']

    return assigned_clustered_df


def get_column_values_by_type(single_bin, column_type):
    if column_type == 'result':
        return single_bin.get_binned_result_column()
    elif column_type == 'source':
        return single_bin.get_binned_source_column()
    else:
        raise Exception('No such column type')


def assign_multicolumn_probabilites(clustered_df, bins, column_type='result'):
    explain_cols = [get_column_values_by_type(single_bin, column_type).name for single_bin in bins]
    binned_explain_cols = [explain_col + '_BINNED_' + column_type.upper() for explain_col in explain_cols]

    assigned_clustered_df = clustered_df.copy()
    proba_vals_df = None

    for index, single_bin in enumerate(bins):
        if proba_vals_df is None:
            proba_vals_df = pd.DataFrame(get_column_values_by_type(single_bin, column_type))
        else:
            proba_vals_df = proba_vals_df.join(get_column_values_by_type(single_bin, column_type))
        # here proba_vals_df contains only rows corresponding to the clustered data if column_type is 'result'

        value_bins = list(np.unique(get_column_values_by_type(single_bin, column_type).values))

        # binning
        explain_col = explain_cols[index]
        binned_explain_col = binned_explain_cols[index]
        assigned_clustered_df[binned_explain_col] = assigned_clustered_df[explain_col]
        if type(value_bins[0]) != str:
            for val_bin in value_bins:
                assigned_clustered_df[binned_explain_col] = assigned_clustered_df.apply(
                    lambda x: val_bin if x[explain_col] > val_bin else x[binned_explain_col], axis=1)

    # assigning each bin its probability
    proba_vals_df.rename(columns={explain_cols[i]: binned_explain_cols[i] for i in range(len(explain_cols))},
                         inplace=True)
    proba_vals = proba_vals_df.value_counts(normalize=True)

    binned_proba = pd.DataFrame(proba_vals, columns=['VAL_PROBA']).reset_index()

    # assigning each row its probability
    assigned_clustered_df = assigned_clustered_df.merge(binned_proba, on=binned_explain_cols, how='left').fillna(0.0)

    # assigning the A function
    assigned_clustered_df['A_VAL_' + column_type.upper()] = assigned_clustered_df['VAL_PROBA']
    assigned_clustered_df.drop(columns=['VAL_PROBA'] + binned_explain_cols, inplace=True)
    assigned_clustered_df.set_index(clustered_df.index, inplace=True)

    return assigned_clustered_df


def local_sufficiency2(df, row, explanation_probas_sum, explanation_proba_col):
    return df[df['cluster'] == int(row['cluster'])][explanation_proba_col].sum() / explanation_probas_sum


def cohort_sufficiency2(df, cluster_id, explanation_proba_col):
    explanation_probas_sum = df[explanation_proba_col].sum()
    df[f'cluster_{cluster_id}'] = df['cluster'].apply(lambda x: 1 if x == cluster_id else 0)
    df[f'cluster_{cluster_id}_multiplied'] = df[f'cluster_{cluster_id}'] * df[explanation_proba_col]
    cluster_sufficiency2 = df[f'cluster_{cluster_id}_multiplied'].sum() / explanation_probas_sum
    cluster_size = len(df[df['cluster'] == cluster_id].index)
    return cluster_sufficiency2 * cluster_size, cluster_size


def sim_from_dist(distances, conversion_type='gaussian', mean_distance=0, std_distance=4):
    if conversion_type == 'reciprocal':
        return 1 / (1 + distances)
    if conversion_type == 'gaussian':
        return np.exp(- (distances - mean_distance) ** 2 / (2 * std_distance ** 2))


def local_sufficiency_with_similarity3(df, embedding_row, embeddings, explanation_probas_sum, explanation_proba_col,
                                       conversion_type='gaussian', mean_distance=0, std_distance=4):
    distances = np.linalg.norm(embeddings.to_numpy() - embedding_row.to_numpy(), axis=1)
    similarities = sim_from_dist(distances, conversion_type, mean_distance, std_distance)
    return np.inner(similarities, df[explanation_proba_col].to_numpy()) / explanation_probas_sum


def cohort_sufficiency_with_similarity3(df, cluster_id, explanation_proba_col, df_embeddings,
                                        sim_conversion_type='gaussian', mean_distance=0, std_distance=4):
    explanation_probas_sum = df[explanation_proba_col].sum()
    cluster_df = df[df['cluster'] == cluster_id]
    cluster_size = len(cluster_df.index)
    cluster_embeddings = cluster_df[df_embeddings.columns]
    sum_local_sufficiencies = sum([local_sufficiency_with_similarity3(df, cluster_embeddings.loc[row_id], df_embeddings,
                                                                      explanation_probas_sum, explanation_proba_col,
                                                                      sim_conversion_type, mean_distance, std_distance)
                                   for row_id in cluster_df.index])
    return sum_local_sufficiencies, cluster_size


def local_sufficiency_with_similarity4(df, embedding_row, embeddings, explanation_probas_sum, explanation_proba_col):
    distances = np.linalg.norm(embeddings.to_numpy() - embedding_row.to_numpy(), axis=1)
    similarities = sim_from_dist(distances)
    return np.inner(similarities, df[explanation_proba_col].to_numpy()) / explanation_probas_sum


def cohort_sufficiency_with_similarity4(df, cluster_id, explanation_proba_col, df_embeddings):
    explanation_probas_sum = df[explanation_proba_col].sum()
    cluster_df = df[df['cluster'] == cluster_id]
    cluster_size = len(cluster_df.index)
    cluster_embeddings = cluster_df[df_embeddings.columns]
    sum_local_sufficiencies = sum([local_sufficiency_with_similarity4(cluster_df, cluster_embeddings.loc[row_id],
                                                                      cluster_embeddings, explanation_probas_sum,
                                                                      explanation_proba_col) for row_id in
                                   cluster_df.index])
    return sum_local_sufficiencies, cluster_size


def mean_distance_from_centroid(centroid, embeddings):
    distances = np.linalg.norm(embeddings.to_numpy() - centroid)
    similarities = sim_from_dist(distances)
    return np.average(distances), np.average(similarities)


def normalize_by_clusters_num(raw_sufficiency, num_clusters):
    return (raw_sufficiency - 1.0 / num_clusters) / (1.0 - 1.0 / num_clusters)


def get_proba_from_bin(bin):
    return pd.concat([bin.get_binned_source_column().value_counts(normalize=True).rename('origin'),
                      bin.get_binned_result_column().value_counts(normalize=True).rename('cluster_binned')],
                     axis=1).fillna(0.0)['cluster_binned']


def contract_odd_indices(series):
    series = series.sort_index(ascending=True)
    series_length = len(series)
    new_series = pd.Series(name=series.name)
    modulu_value = 0  # used to determine if contraction is done on odd or even indices.
    if series_length % 2 == 1:
        new_series = pd.Series(series.iloc[0], index=[series.index[0]], name=series.name)
        modulu_value = 1
    for index_num in range(len(series)):
        if index_num % 2 == modulu_value:
            new_series = pd.concat([new_series, pd.Series(series.iloc[index_num] + series.iloc[index_num + 1],
                                                          index=[series.index[index_num]])])
    return new_series


def get_same_index_series(proba_series1, proba_series2):
    if len(proba_series1) == len(proba_series2):
        return proba_series1, proba_series2
    if len(proba_series1) > len(proba_series2):
        return contract_odd_indices(proba_series1), proba_series2
    return proba_series1, contract_odd_indices(proba_series2)


def diversity_measure(current_bins, new_bins):
    diversity_addition = 0
    for new_bin in new_bins:
        bin_name = new_bin.get_binned_result_column().name
        if bin_name in current_bins:
            js_divs = []
            for current_bin in current_bins[bin_name]:
                new_bin_series, current_bin_series = get_same_index_series(get_proba_from_bin(new_bin),
                                                                           get_proba_from_bin(current_bin))
                try:
                    js_divs.append(wasserstein_distance(new_bin_series, current_bin_series))
                except Exception:
                    js_divs.append(ks_2samp(new_bin_series, current_bin_series).statistic)
            diversity_addition += min(js_divs)
            current_bins[bin_name].append(new_bin)
        else:
            diversity_addition += 1
            current_bins[bin_name] = [new_bin]
    return current_bins, diversity_addition


def cluster_metrics(clustered_dataset, embeddings, cluster_id, suff_configurations, n_clusters,
                    dataset_name="dataset", multicolumm_proba=True):
    print(f'Explanation for cluster {cluster_id}:')
    results, scores = filter_(clustered_dataset.drop(list(embeddings.columns), axis=1), dataset_name, "cluster",
                              "==", cluster_id, ignore={}, to_display=False)
    bins = unzip_bins(results)

    if multicolumm_proba:
        clustered_dataset_with_probas = assign_multicolumn_probabilites(clustered_dataset, bins)
    else:
        clustered_dataset_with_probas = assign_probabilites(clustered_dataset, bins[0])
    cluster_cohort_sufficiencies = [0 for _ in suff_configurations]
    for indx, conf in enumerate(suff_configurations):
        if conf['type'] == 3:
            try:
                cluster_cohort_sufficiency, cluster_size = cohort_sufficiency_with_similarity3(
                    clustered_dataset_with_probas, cluster_id, 'A_VAL_RESULT', embeddings, conf['conversion_type'],
                    conf['mean_distance'], conf['std_distance'])
            except Exception:
                cluster_cohort_sufficiency, cluster_size = cohort_sufficiency_with_similarity3(
                    clustered_dataset_with_probas, cluster_id, 'A_VAL_RESULT', embeddings, conf['conversion_type'])
            cluster_cohort_sufficiency /= cluster_size
        else:
            cluster_cohort_sufficiency, cluster_size = cohort_sufficiency2(clustered_dataset_with_probas,
                                                                           cluster_id, 'A_VAL_RESULT')
            cluster_cohort_sufficiency /= cluster_size
            cluster_cohort_sufficiency = normalize_by_clusters_num(cluster_cohort_sufficiency, n_clusters)
        cluster_cohort_sufficiencies[indx] = cluster_cohort_sufficiency

    cluster_kl_score = np.average([scores[single_bin.get_binned_result_column().name] for single_bin in bins])
    return cluster_cohort_sufficiencies, cluster_kl_score, cluster_size, bins


def explanation_metrics(num_clusters, dataset, embeddings, suff_configurations, dataset_name="dataset",
                        multicolumm_proba=True, random_state=0):
    print(f"Evaluating {num_clusters} clusters")
    num_of_conf = len(suff_configurations)
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(embeddings)
    clustered_dataset = dataset.join(pd.DataFrame(data=clusters, columns=['cluster'], index=dataset.index)).join(
        embeddings)
    total_suff = [0 for conf in suff_configurations]
    per_cluster_suffs = [[] for conf in suff_configurations]
    total_size = 0
    total_diversity = 0
    sizes = []
    all_bins = {}

    mean_kl_score = 0
    for cluster_id in range(num_clusters):
        cluster_cohort_sufficiencies, cluster_kl_score, cluster_size, cluster_bins = cluster_metrics(clustered_dataset,
                                                                                                     embeddings,
                                                                                                     cluster_id,
                                                                                                     suff_configurations,
                                                                                                     num_clusters,
                                                                                                     dataset_name,
                                                                                                     multicolumm_proba)

        for indx in range(num_of_conf):
            per_cluster_suffs[indx].append(cluster_cohort_sufficiencies[indx])
            total_suff[indx] += cluster_cohort_sufficiencies[indx] * cluster_size
        all_bins, cluster_diversity_addition = diversity_measure(all_bins, cluster_bins)
        total_diversity += cluster_diversity_addition
        mean_kl_score += cluster_kl_score
        total_size += cluster_size
        sizes.append(cluster_size)

    # Per hyperparam definitions and aggregations
    mean_kl_score = mean_kl_score / num_clusters
    # mean_dist = mean_dist / N_CLUSTERS
    # mean_sim = mean_sim / N_CLUSTERS
    sil_score = silhouette_score(clustered_dataset[embeddings.columns], clustered_dataset['cluster'])
    ch_score = calinski_harabasz_score(clustered_dataset[embeddings.columns], clustered_dataset['cluster'])
    mean_suffs = np.asarray(total_suff) / total_size
    macro_mean_suffs = [0 for _ in range(num_of_conf)]
    for indx in range(num_of_conf):
        macro_mean_suffs[indx] = np.average(per_cluster_suffs[indx])
    mean_cluster_size = np.average(sizes)
    results_dict = {
        "mean_suffs": mean_suffs,
        "macro_mean_suffs": macro_mean_suffs,
        "mean_kl_score": mean_kl_score,
        "sil_score": sil_score,
        "ch_score": ch_score,
        "mean_cluster_size": mean_cluster_size,
        "per_cluster_suffs": per_cluster_suffs,
        "total_diversity": total_diversity
    }
    return results_dict


def metrics_from_dataset(dataset, embeddings,
                         hyperparam_range=range(2, 50, 2), multicolumm_proba=True,
                         dataset_name="dataset"):
    # distances = pairwise_distances(embeddings.sample(min(len(embeddings.index), 10000)))
    suff_configurations = [
        {'type': 2, 'conversion_type': 'reciprocal'}
        # {'type': 3, 'conversion_type': 'gaussian', 'mean_distance': 0, 'std_distance': distances.mean()}
    ]
    hyperparams_size = len(hyperparam_range)
    knee_conf_index = 1
    num_of_conf = len(suff_configurations)
    js = []
    total_mean_suffs_list = [[] for _ in range(num_of_conf)]
    total_macro_mean_suffs_list = [[] for _ in range(num_of_conf)]
    total_mean_cluster_sizes = []
    total_sil_scores = []
    total_ch_scores = []
    total_mean_kl_scores = []
    total_diversities = []
    total_per_cluster_suffs = [[] for _ in range(num_of_conf)]
    js = list(hyperparam_range)

    pool = ParallelPool(nodes=10)
    results_dicts = pool.map(explanation_metrics, js, [dataset] * hyperparams_size, [embeddings] * hyperparams_size,
                             [suff_configurations] * hyperparams_size, [dataset_name] * hyperparams_size,
                             [multicolumm_proba] * hyperparams_size)

    for result_dict in results_dicts:
        # Total definitions and aggregations
        for indx in range(num_of_conf):
            total_mean_suffs_list[indx].append(result_dict["mean_suffs"][indx])
            total_macro_mean_suffs_list[indx].append(result_dict["macro_mean_suffs"][indx])
            total_per_cluster_suffs[indx].append(result_dict["per_cluster_suffs"][indx])
        total_sil_scores.append(result_dict["sil_score"])
        total_ch_scores.append(result_dict["ch_score"])
        total_mean_kl_scores.append(result_dict["mean_kl_score"])
        total_mean_cluster_sizes.append(result_dict["mean_cluster_size"])
        total_diversities.append(result_dict["total_diversity"])

    # if knee_conf_index:
    #     best_param = KneeLocator(js, gaussian_filter1d(total_macro_mean_suffs_list[knee_conf_index], 1),
    #                              curve='concave').knee
    # else:
    #     best_param = 0
    #     for conf_id in range(num_of_conf):
    #         best_param += KneeLocator(js, gaussian_filter1d(total_mean_suffs_list[conf_id], 1), curve='concave').knee
    #     best_param = np.round(best_param / num_of_conf)

    metrics = {"js": js,
               "mean_suffs": total_mean_suffs_list,
               "per_cluster_suffs": total_per_cluster_suffs,
               "macro_mean": total_macro_mean_suffs_list,
               "sil_scores": total_sil_scores,
               "ch_scores": total_ch_scores,
               "mean_kl_scores": total_mean_kl_scores,
               "mean_cluster_sizes": total_mean_cluster_sizes,
               "diversities": total_diversities,
               # "best_param": best_param
               }
    return metrics


def explain_cluster(clustered_dataset, cluster_id, dataset_name='dataset', display_explanation=True):
    return filter_(clustered_dataset, dataset_name, "cluster", "==", cluster_id, ignore={},
                   to_display=display_explanation)


def cluster_dataset(dataset, embeddings, n_clusters, random_seed=0, return_model=False):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
    clusters = kmeans.fit_predict(embeddings)
    clustered_dataset = dataset.join(pd.DataFrame(data=clusters, columns=['cluster'], index=embeddings.index))
    # clustered_dataset = clustered_dataset.sort_index()
    if return_model:
        return clustered_dataset, kmeans
    return clustered_dataset


def get_bins_per_cluster(clustered_dataset, n_clusters):
    bins_list = []
    for cluster_id in range(n_clusters):
        results, scores = filter_(clustered_dataset, "", "cluster", "==", cluster_id, ignore={}, to_display=False)
        bins_list.append(unzip_bins(results))
    return bins_list


def assign_explanation_probabilities(clustered_dataset, n_clusters, dataset_to_assign, multicolumn_proba=True):
    assert n_clusters > 0
    bins_list = get_bins_per_cluster(clustered_dataset, n_clusters)
    final_assigned_df = None
    for cluster_id in range(n_clusters):
        if multicolumn_proba:
            assigned_df = assign_multicolumn_probabilites(dataset_to_assign, bins_list[cluster_id])[
                list(dataset_to_assign.columns) + ['A_VAL_RESULT']]
            assigned_df = assigned_df.join(
                assign_multicolumn_probabilites(dataset_to_assign, bins_list[cluster_id], 'source')[['A_VAL_SOURCE']],
                how='left')
        else:
            assigned_df = assign_probabilites(clustered_dataset, bins_list[cluster_id][0])
        assigned_df[f'LIFT_{cluster_id}'] = assigned_df.apply(
            lambda row: 0.0 if row['A_VAL_SOURCE'] == 0.0 else row['A_VAL_RESULT'] / row['A_VAL_SOURCE'], axis=1)
        assigned_df = assigned_df.drop(columns=['A_VAL_RESULT', 'A_VAL_SOURCE'])
        if final_assigned_df is None:
            final_assigned_df = assigned_df
        else:
            final_assigned_df = final_assigned_df.join(assigned_df[[f'LIFT_{cluster_id}']])
    return final_assigned_df


def assign_clusters_by_explanation(clustered_train, test_embeddings, test_dataset, n_clusters, kmeans_train_model):
    assigned_test = assign_explanation_probabilities(clustered_train, n_clusters, test_dataset)
    kmeans_assigned_test = pd.Series(kmeans_train_model.predict(test_embeddings), index=test_embeddings.index,
                                     name='clustering_pred')
    assigned_test = assigned_test.join(kmeans_assigned_test)
    assigned_candidates = assigned_test[[f'LIFT_{cluster_id}' for cluster_id in range(n_clusters)]]
    argmax_assignee = assigned_candidates.idxmax(axis=1).apply(lambda x: int(x.replace('LIFT_', '')))
    assigned_test = assigned_test.join(pd.Series(argmax_assignee, index=assigned_test.index, name='explanation_pred'))
    return assigned_test


def evaluate_explanation_by_predictions(train_dataset, test_dataset, train_embeddings, test_embeddings, n_clusters,
                                        metric_type='macro_f1'):
    for dataset in [train_dataset, test_dataset, train_embeddings, test_embeddings]:
        dataset.sort_index(inplace=True)
    clustered_train, kmeans_train = cluster_dataset(train_dataset, train_embeddings, n_clusters, return_model=True)
    assigned_test = assign_clusters_by_explanation(clustered_train, test_embeddings, test_dataset, n_clusters,
                                                   kmeans_train)
    if metric_type == 'report':
        print(classification_report(assigned_test['clustering_pred'], assigned_test['explanation_pred']))
    elif metric_type == 'macro_f1':
        return f1_score(assigned_test['clustering_pred'], assigned_test['explanation_pred'], average='macro')
    elif metric_type == 'accuracy':
        return accuracy_score(assigned_test['clustering_pred'], assigned_test['explanation_pred'])
    return
