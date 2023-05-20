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
                spearman_corr = float(spearmanr(fit_by_exp_scores, metrics_scores).statistic)
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
