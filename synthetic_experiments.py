# Path hack.
import sys, os

sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import time
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from kneed import KneeLocator
from sklearn.model_selection import train_test_split
from explanations_utils import *
import json
from tqdm import tqdm


def generate_synthetic_embeddings(dim, n_clusters, to_print=False):
    if to_print:
        print(f"dim={dim}, n_clusters={n_clusters}")
    features, clusters = make_blobs(n_samples=10000,
                                    n_features=int(dim),
                                    centers=n_clusters,
                                    cluster_std=1.5,
                                    shuffle=True)
    synthetic_dataset = pd.DataFrame(features)
    synthetic_dataset.rename(columns={column: str(column) for column in synthetic_dataset.columns}, inplace=True)
    embedding_dataset = synthetic_dataset.rename(
        columns={column: column + '_emb' for column in synthetic_dataset.columns}, inplace=False)
    return synthetic_dataset, embedding_dataset


def single_synthetic_experiment(dim, n_clusters, to_print=False, bayes=False):
    synthetic_dataset, embedding_dataset = generate_synthetic_embeddings(dim, n_clusters, to_print=to_print)
    metrics_dict = metrics_from_dataset(synthetic_dataset, embedding_dataset,
                                        hyperparam_range=range(3, min(2 * n_clusters + 5, 35), 1), bayes=bayes)
    total_score = MinMaxScaler().fit_transform(
        np.array(metrics_dict['diversities']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['mean_kl_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['sil_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['macro_mean'][0]).reshape(-1, 1))
    return int(metrics_dict['js'][np.argmax(total_score)])


def synthetic_metric_experiment(dim, num_experiments=10, bayes=False):
    single_results_mean = {}
    single_results_std = {}
    for n_clusters in range(5, 21, 5):
        print(f"dim={dim}, n_clusters={n_clusters}")
        max_values = []
        for _ in range(num_experiments):
            max_values.append(single_synthetic_experiment(int(dim), n_clusters, bayes=bayes))
        mean_val = np.mean(max_values)
        std_val = np.std(max_values)
        single_results_mean[n_clusters] = mean_val
        single_results_std[n_clusters] = std_val
        print(f"mean_val={mean_val}, std_val={std_val}")
    return single_results_mean, single_results_std


def synthetic_fit_by_explanation_experiment(dim):
    single_best_results = {}
    single_ranks = {}
    f1_scores = {}
    best_f1_scores = {}
    for n_clusters in range(5, 21, 5):
        synthetic_dataset, embedding_dataset = generate_synthetic_embeddings(dim, n_clusters, to_print=True)
        n_clusters_list = list(range(3, min(2 * n_clusters + 5, 35), 1))
        scores = eval_range_of_fit_by_explanation(synthetic_dataset, None, embedding_dataset, None,
                                                  n_clusters_range=n_clusters_list)
        single_best_results[n_clusters] = int(n_clusters_list[np.argmax(scores)])
        n_clusters_in_clusters_list = find_index(n_clusters_list, n_clusters)
        f1_scores[n_clusters] = scores[n_clusters_in_clusters_list]
        best_f1_scores[n_clusters] = np.max(scores)
        single_ranks[n_clusters] = len(scores) - find_index(sorted(scores), scores[n_clusters_in_clusters_list])
        # note that this way the max rank is 1 and not 0
        print(
            f"best_n={single_best_results[n_clusters]}, best_score={best_f1_scores[n_clusters]}, "
            f"gt_rank={single_ranks[n_clusters]}, gt_score={f1_scores[n_clusters]}"
        )
        del synthetic_dataset, embedding_dataset
    return single_best_results, single_ranks, f1_scores, best_f1_scores


if __name__ == '__main__':
    files_path = '/specific/disk1/home/ronycopul/Projects/TabEE'
    dims_range = np.nditer(np.logspace(1, 2.5, 5))
    bayes = True
    bayes_suffix = 'bayes' if bayes else ''

    results_explanations_best = {}
    results_explanations_rank = {}
    results_explanations_score = {}
    results_explanations_best_score = {}
    for dim in tqdm(dims_range):
        single_best_results, single_ranks, single_scores, single_best_scores = synthetic_fit_by_explanation_experiment(
            int(dim))
        results_explanations_best[int(dim)] = single_best_results
        with open(f'{files_path}/synthetic_results_explanations_best_{bayes_suffix}.json', 'w') as fp:
            json.dump(results_explanations_best, fp)
            print(f"done writing results_best, dim={dim}")

        results_explanations_best_score[int(dim)] = single_best_scores
        with open(f'{files_path}/synthetic_results_explanations_best_score_{bayes_suffix}.json', 'w') as fp:
            json.dump(results_explanations_best_score, fp)
            print(f"done writing results_rank, dim={dim}")

        results_explanations_rank[int(dim)] = single_ranks
        with open(f'{files_path}/synthetic_results_explanations_rank_{bayes_suffix}.json', 'w') as fp:
            json.dump(results_explanations_rank, fp)
            print(f"done writing results_rank, dim={dim}")

        results_explanations_score[int(dim)] = single_scores
        with open(f'{files_path}/synthetic_results_explanations_score_{bayes_suffix}.json', 'w') as fp:
            json.dump(results_explanations_score, fp)
            print(f"done writing results_rank, dim={dim}")
    # results_mean = {}
    # results_std = {}
    # for dim in tqdm(dims_range):
    #     dim = int(dim)
    #     print(f"dim={dim}")
    #     single_results_mean, single_results_std = synthetic_metric_experiment(dim, bayes=bayes)
    #     results_mean[dim] = single_results_mean
    #     results_std[dim] = single_results_std
    #     with open(f'{files_path}/synthetic_results_mean_{bayes_suffix}.json', 'w') as fp:
    #         json.dump(results_mean, fp)
    #         print(f"done writing results_mean, dim={dim}")
    #     with open(f'{files_path}/synthetic_results_std_{bayes_suffix}.json', 'w') as fp:
    #         json.dump(results_std, fp)
    #         print(f"done writing results_std, dim={dim}")
