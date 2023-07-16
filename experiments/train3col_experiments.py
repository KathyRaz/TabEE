import numpy as np
import pandas as pd
import os
from itertools import product
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from embedding_explainer import EmbeddingExplainer
import utils
from custom_classifiers import VIMECustomClassifier, TabNetCustomClassifier


def train_3col_experiment(name, dataset, target_col):
    for cols_list in [random.sample(dataset.drop(columns=[target_col]), 3) for _ in range(5)]:


def single_train_3col_vime(dataset, cols_list, num_cols, cat_cols, target_col):
    higgs_vime_classifier = VIMECustomClassifier(name="higgs_vime_sample", num_cols=num_cols, cat_cols=cat_cols,
                                                   target_col=target_col)
    higgs_vime_classifier.fit(dataset[cols_list], dataset[target_col])
    embedding_vime = pd.DataFrame(higgs_vime_classifier.get_embedding(dataset)[cols_list], index=dataset.index)
    higgs_vime_explainer = EmbeddingExplainer(dataset, embedding_vime, range(2, 8, 1), diversity_weight=0.05)
    higgs_vime_explainer.choose_n_clusters_by_explanation(3)


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

    covtype_sample = covtype_test.sample(20000, random_state=42)

    combinations = [
        {
            "name": "covtype tabnet",
            "dataset": covtype_dataset.loc[covtype_sample.index],
            "embedding": covtype_test_tabnet.loc[covtype_sample.index]
        },
        {
            "name": "covtype vime",
            "dataset": covtype_dataset.loc[covtype_sample.index],
            "embedding": covtype_test_vime.loc[covtype_sample.index]
        },
        {
            "name": "covtype transtab",
            "dataset": covtype_dataset.loc[covtype_sample.index],
            "embedding": covtype_test_transtab.loc[covtype_sample.index]
        },
        {
            "name": "spotify tabnet",
            "dataset": spotify_dataset.loc[spotify_test.index],
            "embedding": spotify_test_tabnet
        },
        {
            "name": "spotify vime",
            "dataset": spotify_dataset.loc[spotify_test.index],
            "embedding": spotify_test_vime
        },
        {
            "name": "spotify transtab",
            "dataset": spotify_dataset.loc[spotify_test.index],
            "embedding": spotify_test_transtab
        }]

    for combination in combinations:
        fit_by_explanation(combination["name"] + " low div", combination["dataset"], combination["embedding"],
                           diversity_weight=0.01, explanation_score_type='best')
        # fit_by_explanation(combination["name"] + " no div", combination["dataset"], combination["embedding"],
        #                    diversity_weight=0.0, explanation_score_type='best')
        # fit_by_explanation(combination["name"] + " no sil", combination["dataset"], combination["embedding"],
        #                    diversity_weight=0.05, sil_weight=0, explanation_score_type='best')
        # fit_by_explanation(combination["name"] + " no int", combination["dataset"], combination["embedding"],
        #                    diversity_weight=1.0, interest_weight=0, explanation_score_type='best')
        # fit_by_explanation(combination["name"] + " no suff", combination["dataset"], combination["embedding"],
        #                    diversity_weight=1.0, suff_weight=0, explanation_score_type='best')
    # fit_by_explanation("covtype tabnet best", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_tabnet.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='best')

    # fit_by_explanation("covtype tabnet median", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_tabnet.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='median')
    #
    # fit_by_explanation("covtype tabnet worst", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_tabnet.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='worst')

    # fit_by_explanation("covtype vime best", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_vime.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='best')

    # fit_by_explanation("covtype vime median", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_vime.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='median')
    #
    # fit_by_explanation("covtype vime worst", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_vime.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='worst')

    # fit_by_explanation("covtype transtab best", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_transtab.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='best')

    # fit_by_explanation("covtype transtab median", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_transtab.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='median')
    #
    # fit_by_explanation("covtype transtab worst", covtype_dataset.loc[covtype_sample.index],
    #                    covtype_test_transtab.loc[covtype_sample.index],
    #                    diversity_weight=0.05, explanation_score_type='worst')

    # fit_by_explanation("spotify tabnet best", spotify_dataset.loc[spotify_test.index], spotify_test_tabnet,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='best')

    # fit_by_explanation("spotify tabnet median", spotify_dataset.loc[spotify_test.index], spotify_test_tabnet,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='median')
    #
    # fit_by_explanation("spotify tabnet worst", spotify_dataset.loc[spotify_test.index], spotify_test_tabnet,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='worst')

    # fit_by_explanation("spotify vime best", spotify_dataset.loc[spotify_test.index], spotify_test_vime,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='best')

    # fit_by_explanation("spotify vime median", spotify_dataset.loc[spotify_test.index], spotify_test_vime,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='median')
    #
    # fit_by_explanation("spotify vime worst", spotify_dataset.loc[spotify_test.index], spotify_test_vime,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='worst')

    # fit_by_explanation("spotify transtab best", spotify_dataset.loc[spotify_test.index], spotify_test_transtab,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='best')

    # fit_by_explanation("spotify transtab median", spotify_dataset.loc[spotify_test.index], spotify_test_transtab,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='median')
    #
    # fit_by_explanation("spotify transtab worst", spotify_dataset.loc[spotify_test.index], spotify_test_transtab,
    #                    diversity_weight=0.05,
    #                    explanation_score_type='worst')
