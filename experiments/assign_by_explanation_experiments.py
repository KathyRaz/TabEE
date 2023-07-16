import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from embedding_explainer import EmbeddingExplainer
import utils


def fit_by_explanation(name, dataset, embedding, sil_weight=1,
                       suff_weight=1, interest_weight=1, diversity_weight=0.05, clustering_method='kmeans', explanation_score_type='best'):
    print(name)
    explainer = EmbeddingExplainer(dataset, embedding, range(2, 8, 1), sil_weight=1,
                                   suff_weight=1, interest_weight=1, diversity_weight=diversity_weight,
                                   clustering_method=clustering_method,
                                   explanation_score_type=explanation_score_type)
    explainer.choose_n_clusters_by_explanation(5)
    clusters_probas = explainer.final_clustered_dataset.cluster.value_counts(normalize=True)
    print(explainer.n_clusters)
    score = np.average(
        [single_fit_by_explanation(dataset, embedding, explainer, clusters_probas) for _ in range(10000)])
    print(score)
    print(utils.normalize_by_clusters_num(score, explainer.n_clusters))


def single_fit_by_explanation(dataset, embedding, explainer, cluster_probas):
    random_state = np.random.randint(10000)
    single_sample = dataset.loc[embedding.sample(2, random_state=random_state).index[1]].to_frame().T
    real_embedding = embedding.loc[embedding.sample(2, random_state=random_state).index[1]].to_frame().T

    real_cluster_id = int(explainer.clustering_model.predict(real_embedding))

    probas_list = []
    for cluster_id in range(explainer.n_clusters):
        real_explain_col_name, real_explain_dist = explainer.embedding_explanation[cluster_id]
        real_proba_lift = utils.distributions_lift(explainer.source_distributions[real_explain_col_name]['dist'],
                                                   real_explain_dist) * cluster_probas[cluster_id]
        if explainer.source_distributions[real_explain_col_name]['data_type'] == 'numeric':
            real_proba_bins = explainer.source_distributions[real_explain_col_name]['bin']
            real_bin = pd.cut(single_sample[real_explain_col_name], bins=real_proba_bins,
                              labels=real_proba_bins[:-1],
                              right=False, duplicates='drop')
        else:
            real_bin = single_sample[real_explain_col_name]
        probas_list.append(real_proba_lift[real_bin.iloc[0]])
    best_cluster = np.argmax(probas_list)

    return real_cluster_id == best_cluster


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
    covtype_test_transtab = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='transtab_test'),
        index_col='Unnamed: 0')

    spotify_test_tabnet = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='tabnet_test'),
        index_col='Unnamed: 0')
    spotify_test_transtab = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='transtab_test'),
        index_col='Unnamed: 0')
    spotify_test_vime = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='vime_test'),
        index_col='Unnamed: 0')
    spotify_test_embdi = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='embdi_test'),
        index_col='Unnamed: 0')
    covtype_test_vime = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='vime_test'),
        index_col='Unnamed: 0')

    covtype_sample = covtype_test.sample(20000, random_state=42)

    higgs_dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='higgs_sample'), index_col='Unnamed: 0')
    higgs_dataset = higgs_dataset.rename(
        columns={col: col+'_col' for col in higgs_dataset.drop(columns=['target']).columns})
    higgs_vime = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='higgs_sample', embedding_name='vime'),
        index_col='Unnamed: 0')
    higgs_tabnet = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='higgs_sample', embedding_name='tabnet'),
        index_col='Unnamed: 0')
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
        },
        {
            "name": "higgs tabnet",
            "dataset": higgs_dataset,
            "embedding": higgs_tabnet
        },
        {
            "name": "higgs vime",
            "dataset": higgs_dataset,
            "embedding": higgs_vime
        },
        {
            "name": "spotify embdi",
            "dataset": spotify_dataset.loc[spotify_test.index],
            "embedding": spotify_test_embdi
        },
        ]

    for combination in combinations:
        fit_by_explanation(combination["name"] + " default", combination["dataset"], combination["embedding"],
                           diversity_weight=0.05, explanation_score_type='best')
        fit_by_explanation(combination["name"] + " gmm", combination["dataset"], combination["embedding"],
                           diversity_weight=0.05, clustering_method='gmm', explanation_score_type='best')
        # fit_by_explanation(combination["name"] + " low div", combination["dataset"], combination["embedding"],
        #                    diversity_weight=0.01, explanation_score_type='best')
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
