# Path hack.
import sys, os

sys.path.insert(0, os.path.abspath('..'))
import numpy as np
from scipy.stats import spearmanr
import time
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from explanations_utils import *
import json
from tqdm import tqdm

_DIRECTORY_PATH = '/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/'
_EMBEDDINGS_PATH_FORMAT = _DIRECTORY_PATH + 'embeddings/{dataset_name}_embeddings_{embedding_name}.csv'
_DATASETS_PATH_FORMAT = _DIRECTORY_PATH + 'datasets/{dataset_name}.csv'


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
        scores = eval_range_of_fit_by_explanation(synthetic_dataset, None, embedding_dataset, None,
                                                  n_clusters_range=n_clusters_list)
        single_best_results[n_clusters] = int(n_clusters_list[np.argmax(scores)])
        n_clusters_in_clusters_list = find_index(n_clusters_list, n_clusters)
        single_ranks[n_clusters] = len(scores) - find_index(sorted(scores), scores[n_clusters_in_clusters_list])
        print(f"best_n={single_best_results[n_clusters]}, rank={single_ranks[n_clusters]}")
        del features, clusters, synthetic_dataset, embedding_dataset
    return single_best_results, single_ranks


def fit_by_explanation_experiments(dataset_names, embedding_names, sample_sizes={}):
    n_clusters_list = list(range(3, 25, 1))
    experiments_results = {}
    for dataset_name in dataset_names:
        experiments_results[dataset_name] = {}
        for embedding_name in embedding_names:
            try:
                embedding = pd.read_csv(
                    _EMBEDDINGS_PATH_FORMAT.format(dataset_name=dataset_name, embedding_name=embedding_name))
                dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name=dataset_name))
                if dataset_name in sample_sizes.keys():
                    embedding = embedding.sample(sample_sizes[dataset_name], random_state=0)
                    dataset = dataset.sample(sample_sizes[dataset_name], random_seed=0)
                print(f"read {dataset_name}_{embedding_name} successfully")
                fit_by_exp_scores = eval_range_of_fit_by_explanation(dataset, None, embedding, None, n_clusters_list)
                max_n_fit_by_exp = int(n_clusters_list[np.argmax(np.array(fit_by_exp_scores))])
                metrics_scores = explanation_scores_from_dataset(dataset, embedding)
                max_metric_n_index = np.argmax(np.array(metrics_scores))
                max_n_metrics = int(n_clusters_list[max_metric_n_index])
                spearman_corr = float(spearmanr(fit_by_exp_scores, metrics_scores))
                rank_in_fit_by_exp = len(fit_by_exp_scores) - find_index(sorted(fit_by_exp_scores),
                                                                         fit_by_exp_scores[max_metric_n_index])
                experiments_results[dataset_name][embedding_name] = {'metric_max': max_n_metrics,
                                                                     'fit_by_exp_max': max_n_fit_by_exp,
                                                                     'metric_max_rank_in_fit_by_exp': rank_in_fit_by_exp,
                                                                     'spearman_corr': spearman_corr}
                print(f"done {dataset_name}_{embedding_name} exepriment")
            except Exception as e:
                print(f"Error reading {dataset_name}_{embedding_name}: {e}")
    return experiments_results


if __name__ == '__main__':
    dataset_names = ['flights', 'covtype', 'higgs', 'spotify']
    embedding_names = ['tabnet', 'vime', 'transtab']
    sample_sizes = {'higgs': 500000, 'flights': 500000}
    experiments_results = fit_by_explanation_experiments(dataset_names, embedding_names, sample_sizes)

    with open('/specific/disk1/home/ronycopul/Projects/Explain_tab_emb/datasets_experiments_results.json', 'w') as fp:
        json.dump(experiments_results, fp)
        print(f"done writing experiments_results")
