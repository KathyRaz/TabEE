import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["NUM_THREADS"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
from embedding_explainer import EmbeddingExplainer
import utils


def runtime_experiment(dataset, embedding, num_candidates, max_n):
    max_n = int(min(max_n, 24 * np.log(2) / np.log(num_candidates)))
    explainer = EmbeddingExplainer(dataset, embedding, range(2, max_n + 1, 1), diversity_weight=0.05, random_state=1)
    return max_n, *explainer.choose_n_clusters_by_explanation(num_candidates, eval_time=True)


if __name__ == '__main__':
    _DIRECTORY_PATH = '/specific/disk1/home/ronycopul/Projects/TabEE/'
    _EMBEDDINGS_PATH_FORMAT = _DIRECTORY_PATH + 'embeddings/{dataset_name}_embeddings_{embedding_name}.csv'
    _DATASETS_PATH_FORMAT = _DIRECTORY_PATH + 'datasets/{dataset_name}.csv'

    spotify_dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='spotify'))
    spotify_test = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='spotify_userstudy_dataset_test'),
                               index_col='Unnamed: 0')

    covtype_dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='covtype_categorical'))

    covtype_test = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='covtype_userstudy_dataset_test'),
                               index_col='Unnamed: 0')

    covtype_test_tabnet = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='tabnet_test'),
        index_col='Unnamed: 0')

    spotify_test_tabnet = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='tabnet_test'),
        index_col='Unnamed: 0')

    spotify_test_vime = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='vime_test'),
        index_col='Unnamed: 0')

    covtype_test_vime = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='vime_test'),
        index_col='Unnamed: 0')

    covtype_sample = covtype_test.sample(20000, random_state=0)
    max_k = 20
    combinations = [
        {"name": "covtype_vime",
         "dataset": covtype_dataset.loc[covtype_sample.index],
         "embedding": covtype_test_vime.loc[covtype_sample.index]},
        {"name": "covtype_tabnet",
         "dataset": covtype_dataset.loc[covtype_sample.index],
         "embedding": covtype_test_tabnet.loc[covtype_sample.index]
         },
        {"name": "spotify_vime",
         "dataset": spotify_dataset.loc[spotify_test.index],
         "embedding": spotify_test_vime.loc[spotify_test.index]},
        {"name": "covtype_tabnet",
         "dataset": spotify_dataset.loc[spotify_test.index],
         "embedding": spotify_test_tabnet.loc[spotify_test.index]
         }
    ]

    for combination in combinations:
        print(f'{combination["name"]}  experiments')
        results_pd = None
        for num_candidates in range(1, 7, 1):
            print(f"running {num_candidates} candidates")
            max_n, runtime_lists, scores_lists = runtime_experiment(combination["dataset"],
                                                                    combination["embedding"],
                                                                    num_candidates, max_k)
            if results_pd is None:
                results_pd = pd.DataFrame(
                    np.array(
                        [[int(num_candidates)] * (max_n - 1), list(range(2, max_n + 1, 1)), runtime_lists,
                         scores_lists]).T,
                    columns=['num_candidates', 'k_max', 'total_runtime', 'max_score'])
            else:
                results_pd = pd.concat([results_pd, pd.DataFrame(
                    np.array(
                        [[int(num_candidates)] * (max_n - 1), list(range(2, max_n + 1, 1)), runtime_lists,
                         scores_lists]).T,
                    columns=['num_candidates', 'k_max', 'total_runtime', 'max_score'])], ignore_index=True)
            results_pd.to_csv(
                f'/specific/disk1/home/ronycopul/Projects/TabEE/runtime_results_{combination["name"]}.csv')
