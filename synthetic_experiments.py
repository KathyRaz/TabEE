# Path hack.
import sys, os

import os

print(os.path.abspath(__file__))

sys.path.insert(0, os.path.abspath('..'))
# sys.path.append(os.path.abspath('..') + "/FEDEx/")

import numpy as np
import time
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from kneed import KneeLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d

from explanations_utils import *


def graph_from_metrics(metrics_dict):
    plt.plot(metrics_dict['js'], metrics_dict['macro_mean'][0], label="macro mean sufficiency")
    plt.plot(metrics_dict['js'], metrics_dict['sil_scores'], label="silhouette score")
    plt.plot(metrics_dict['js'], metrics_dict['mean_kl_scores'], label='KL div scores')
    plt.plot(metrics_dict['js'], MinMaxScaler().fit_transform(np.array(metrics_dict['diversities']).reshape(-1, 1)),
             label='diversity')
    plt.plot(metrics_dict['js'], MinMaxScaler().fit_transform(
        np.array(metrics_dict['diversities']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['mean_kl_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['sil_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['macro_mean'][0]).reshape(-1, 1)), label='weighted scores')
    print(metrics_dict['js'][np.argmax(MinMaxScaler().fit_transform(
        np.array(metrics_dict['diversities']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['mean_kl_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['sil_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
        np.array(metrics_dict['macro_mean'][0]).reshape(-1, 1)))])
    plt.legend(loc='upper right')
    plt.show()


import json

from tqdm import tqdm


def synthetic_metric_experiment(dim):
    single_results_max = {}
    single_results_knee = {}
    for n_clusters in range(5, 21, 5):
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
        metrics_dict = metrics_from_dataset(synthetic_dataset, embedding_dataset,
                                            hyperparam_range=range(3, min(2 * n_clusters + 5, 35), 1))
        total_score = MinMaxScaler().fit_transform(
            np.array(metrics_dict['diversities']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
            np.array(metrics_dict['mean_kl_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
            np.array(metrics_dict['sil_scores']).reshape(-1, 1)) + MinMaxScaler().fit_transform(
            np.array(metrics_dict['macro_mean'][0]).reshape(-1, 1))
        single_results_max[n_clusters] = int(metrics_dict['js'][np.argmax(total_score)])
        single_results_knee[n_clusters] = int(KneeLocator(np.array(metrics_dict['js']).reshape(-1),
                                                          gaussian_filter1d(total_score, 1).reshape(-1),
                                                          curve='concave').knee)
        print(f"results_max={single_results_max[n_clusters]}, results_knee={single_results_knee[n_clusters]}")
        del metrics_dict, features, clusters, synthetic_dataset, embedding_dataset
    return single_results_max, single_results_knee


def synthetic_fit_by_explanation_experiment(dim):
    single_best_results = {}
    single_ranks = {}
    for n_clusters in range(5, 21, 5):
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
        n_clusters_list = list(range(3, min(2 * n_clusters + 5, 35), 1))
        scores = evaluate_multiple_explanations_by_predictions(synthetic_dataset, None, embedding_dataset, None,
                                                               n_clusters_range=n_clusters_list)
        single_best_results[n_clusters] = int(n_clusters_list[np.argmax(scores)])
        n_clusters_in_clusters_list = find_index(n_clusters_list, n_clusters)
        single_ranks[n_clusters] = find_index(sorted(scores, reverse=True), scores[n_clusters_in_clusters_list])
        print(f"best_n={single_best_results[n_clusters]}, rank={single_ranks[n_clusters]}")
        del features, clusters, synthetic_dataset, embedding_dataset
    return single_best_results, single_ranks


if __name__ == '__main__':
    results_max = {}
    results_knee = {}
    dims_range = np.nditer(np.logspace(1, 2.5, 5))

    results_explanations_best = {}
    results_explanations_rank = {}
    for dim in tqdm(dims_range):
        single_best_results, single_ranks = synthetic_fit_by_explanation_experiment(int(dim))
        results_explanations_best[int(dim)] = single_best_results
        with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_explanations_best.json',
                  'w') as fp:
            json.dump(results_explanations_best, fp)
            print(f"done writing results_best, dim={dim}")

        results_explanations_rank[int(dim)] = single_ranks
        with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_explanations_rank.json',
                  'w') as fp:
            json.dump(results_explanations_rank, fp)
            print(f"done writing results_rank, dim={dim}")

# dim = next(dims_range).item()
# print(f"dim={dim}")
# single_results_max, single_results_knee = synthetic_metric_experiment(dim)
# results_max[dim] = single_results_max
# results_knee[dim] = single_results_knee
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_max.json', 'w') as fp:
#     json.dump(results_max, fp)
#     print(f"done writing results_max, dim={dim}")
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_knee.json', 'w') as fp:
#     json.dump(results_knee, fp)
#     print(f"done writing results_knee, dim={dim}")
#
# dim = next(dims_range).item()
# print(f"dim={dim}")
# single_results_max, single_results_knee = synthetic_metric_experiment(dim)
# results_max[dim] = single_results_max
# results_knee[dim] = single_results_knee
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_max.json', 'w') as fp:
#     json.dump(results_max, fp)
#     print(f"done writing results_max, dim={dim}")
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_knee.json', 'w') as fp:
#     json.dump(results_knee, fp)
#     print(f"done writing results_knee, dim={dim}")
#
# dim = next(dims_range).item()
# print(f"dim={dim}")
# single_results_max, single_results_knee = synthetic_metric_experiment(dim)
# results_max[dim] = single_results_max
# results_knee[dim] = single_results_knee
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_max.json', 'w') as fp:
#     json.dump(results_max, fp)
#     print(f"done writing results_max, dim={dim}")
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_knee.json', 'w') as fp:
#     json.dump(results_knee, fp)
#     print(f"done writing results_knee, dim={dim}")
#
# dim = next(dims_range).item()
# print(f"dim={dim}")
# single_results_max, single_results_knee = synthetic_metric_experiment(dim)
# results_max[dim] = single_results_max
# results_knee[dim] = single_results_knee
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_max.json', 'w') as fp:
#     json.dump(results_max, fp)
#     print(f"done writing results_max, dim={dim}")
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_knee.json', 'w') as fp:
#     json.dump(results_knee, fp)
#     print(f"done writing results_knee, dim={dim}")
#
# dim = next(dims_range).item()
# print(f"dim={dim}")
# single_results_max, single_results_knee = synthetic_metric_experiment(dim)
# results_max[dim] = single_results_max
# results_knee[dim] = single_results_knee
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_max.json', 'w') as fp:
#     json.dump(results_max, fp)
#     print(f"done writing results_max, dim={dim}")
# with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/synthetic_results_knee.json', 'w') as fp:
#     json.dump(results_knee, fp)
#     print(f"done writing results_knee, dim={dim}")
