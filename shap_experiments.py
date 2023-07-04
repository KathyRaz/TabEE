import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from embedding_explainer import EmbeddingExplainer
import os
import multiprocessing as mp
from pathos.pp import ParallelPool

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle
import shap


def shap_values_from_explainer(explainer, input_dataset):
    return explainer(input_dataset).values


def split_dataframe(df, num_chunks=10):
    chunks = list()
    chunk_size = len(df) // num_chunks
    for i in range(num_chunks):
        chunks.append(df.iloc[i * chunk_size:min((i + 1) * chunk_size, len(df))])
    return chunks


def create_cat_cols_mapping(cat_cols):
    def cat_cols_mapping(col_name):
        for col in cat_cols:
            if col in col_name:
                return col
        return col_name

    return cat_cols_mapping


def create_comparison_dataframe(dataset, features_dataset, embeddings, embedding_classifier, cat_cols,
                                n_jobs=1, score_type='best'):
    embedding_explainer = EmbeddingExplainer(dataset, embeddings, n_clusters_range=range(2, 9, 1),
                                             diversity_weight=0.05, explanation_score_type=score_type)
    embedding_explainer.choose_n_clusters_by_explanation(5)

    # Fits the explainer
    shap_explainer = shap.Explainer(embedding_classifier.predict, features_dataset)
    # Calculates the SHAP values
    if n_jobs == 1:
        shap_values = shap_values_from_explainer(shap_explainer, features_dataset)
    else:
        features_dataset_split = split_dataframe(features_dataset)
        with ParallelPool(nodes=n_jobs) as pool:
            result = pool.map(shap_values_from_explainer, [shap_explainer] * n_jobs, features_dataset_split)
        shap_values = np.concatenate(result)
    del shap_explainer

    cat_cols_mapping = create_cat_cols_mapping(cat_cols)

    columns_mapping = {}
    for i in range(len(features_dataset.columns)):
        columns_mapping[i] = list(features_dataset.columns)[i]

    cluster_column_mapping = {}
    for cluster_id in range(embedding_explainer.n_clusters):
        cluster_column_mapping[cluster_id] = embedding_explainer.embedding_explanation[cluster_id][0]
    explain_cols = pd.DataFrame.from_dict(cluster_column_mapping, orient='index', columns=['explain']).reset_index(
        names='cluster')

    abs_values_np = np.abs(
        pd.DataFrame(shap_values, index=features_dataset.index, columns=features_dataset.columns).to_numpy())
    total_condition = None
    accuracies = {}
    for k in range(1, 10, 1):
        shap_columns = pd.Series(
            np.vectorize(columns_mapping.get)(np.argpartition(abs_values_np, -1 * k, axis=1)[:, -1 * k]),
            index=dataset.index, name='SHAP').apply(cat_cols_mapping)
        shap_vs_explain = embedding_explainer.final_clustered_dataset.join(shap_columns).merge(explain_cols,
                                                                                               on='cluster',
                                                                                               how='left').set_index(
            embedding_explainer.final_clustered_dataset.index)
        condition = (shap_vs_explain['explain'] != shap_vs_explain['SHAP'])
        if k == 1:
            total_condition = condition
        else:
            total_condition = total_condition & condition

        total_accuracy = 1.0 - len(shap_vs_explain[total_condition]) / len(shap_vs_explain)

        accuracies[k] = total_accuracy
    return accuracies


if __name__ == '__main__':
    _DIRECTORY_PATH = '/specific/disk1/home/ronycopul/Projects/TabEE/'
    _EMBEDDINGS_PATH_FORMAT = _DIRECTORY_PATH + 'embeddings/{dataset_name}_embeddings_{embedding_name}.csv'
    _DATASETS_PATH_FORMAT = _DIRECTORY_PATH + 'datasets/{dataset_name}.csv'

    spotify_dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='spotify'))
    spotify_dataset = spotify_dataset.drop(columns=['genre'])
    spotify_test = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='spotify_userstudy_dataset_test'),
                               index_col='Unnamed: 0')
    spotify_sample = spotify_test.sample(1000, random_state=3)

    spotify_test_tabnet = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='tabnet_test'),
        index_col='Unnamed: 0')
    spotify_test_transtab = pd.read_csv(
        _EMBEDDINGS_PATH_FORMAT.format(dataset_name='spotify_userstudy', embedding_name='transtab_test'),
        index_col='Unnamed: 0')

    with open('/specific/disk1/home/ronycopul/Projects/TabEE/user_study_vars/spotify_transtab_classifier.pkl',
              'rb') as f:
        spotify_transtab_classifier = pickle.load(f)

    # spotify_transtab_worst_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
    #                                                                 spotify_sample,
    #                                                                 spotify_test_transtab.loc[spotify_sample.index],
    #                                                                 spotify_transtab_classifier, [],
    #                                                                 n_jobs=50,
    #                                                                 score_type='worst')
    spotify_transtab_best_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
                                                                   spotify_sample,
                                                                   spotify_test_transtab.loc[spotify_sample.index],
                                                                   spotify_transtab_classifier, [],
                                                                   n_jobs=10,
                                                                   score_type='best')
    # spotify_transtab_median_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
    #                                                                  spotify_sample,
    #                                                                  spotify_test_transtab.loc[spotify_sample.index],
    #                                                                  spotify_transtab_classifier, [],
    #                                                                  n_jobs=50,
    #                                                                  score_type='median')
    del spotify_transtab_classifier

    print(f"acc_spotify_transtab_best: {spotify_transtab_best_accuracies}")
    # print(f"acc_spotify_transtab_median: {spotify_transtab_median_accuracies}")
    # print(f"acc_spotify_transtab_worst: {spotify_transtab_worst_accuracies}")

    with open('/specific/disk1/home/ronycopul/Projects/TabEE/user_study_vars/spotify_tabnet_classifier.pkl', 'rb') as f:
        spotify_tabnet_classifier = pickle.load(f)

    # spotify_tabnet_worst_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
    #                                                               spotify_sample,
    #                                                               spotify_test_tabnet.loc[spotify_sample.index],
    #                                                               spotify_tabnet_classifier, [],
    #                                                               n_jobs=50,
    #                                                               score_type='worst')
    spotify_tabnet_best_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
                                                                 spotify_sample,
                                                                 spotify_test_tabnet.loc[spotify_sample.index],
                                                                 spotify_tabnet_classifier, [],
                                                                 n_jobs=10,
                                                                 score_type='best')
    # spotify_tabnet_median_accuracies = create_comparison_dataframe(spotify_dataset.loc[spotify_sample.index],
    #                                                                spotify_sample,
    #                                                                spotify_test_tabnet.loc[spotify_sample.index],
    #                                                                spotify_tabnet_classifier, [],
    #                                                                n_jobs=50,
    #                                                                score_type='median')
    del spotify_tabnet_classifier

    print(f"acc_spotify_tabnet_best: {spotify_tabnet_best_accuracies}")
    # print(f"acc_spotify_tabnet_median: {spotify_tabnet_median_accuracies}")
    # print(f"acc_spotify_tabnet_worst: {spotify_tabnet_worst_accuracies}")

    #
    # covtype_dataset = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='covtype_categorical'))
    # covtype_test = pd.read_csv(_DATASETS_PATH_FORMAT.format(dataset_name='covtype_userstudy_dataset_test'),
    #                            index_col='Unnamed: 0')
    #
    # covtype_test_tabnet = pd.read_csv(
    #     _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='tabnet_test'),
    #     index_col='Unnamed: 0')
    # covtype_test_transtab = pd.read_csv(
    #     _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='transtab_test'),
    #     index_col='Unnamed: 0')
    # covtype_test_vime = pd.read_csv(
    #     _EMBEDDINGS_PATH_FORMAT.format(dataset_name='covtype_userstudy', embedding_name='vime_test'),
    #     index_col='Unnamed: 0')
    #
    # covtype_sample = covtype_test.sample(10000, random_state=3)
    #
    # with open('/specific/disk1/home/ronycopul/Projects/TabEE/user_study_vars/covtype_vime_classifier.pkl', 'rb') as f:
    #     covtype_vime_classifier = pickle.load(f)
    #
    # covtype_dataset = covtype_dataset.drop(columns=['Cover_Type'])
    # covtype_vime_worst_accuracies = create_comparison_dataframe(covtype_dataset.loc[covtype_sample.index],
    #                                                             covtype_sample,
    #                                                             covtype_test_vime.loc[covtype_sample.index],
    #                                                             covtype_vime_classifier,
    #                                                             ['Wilderness_Area', 'Soil_Type'],
    #                                                             n_jobs=50,
    #                                                             score_type='worst')
    # covtype_vime_best_accuracies = create_comparison_dataframe(covtype_dataset.loc[covtype_sample.index],
    #                                                            covtype_sample,
    #                                                            covtype_test_vime.loc[covtype_sample.index],
    #                                                            covtype_vime_classifier,
    #                                                            ['Wilderness_Area', 'Soil_Type'],
    #                                                            n_jobs=50,
    #                                                            score_type='best')
    # covtype_vime_median_accuracies = create_comparison_dataframe(covtype_dataset.loc[covtype_sample.index],
    #                                                              covtype_sample,
    #                                                              covtype_test_vime.loc[covtype_sample.index],
    #                                                              covtype_vime_classifier,
    #                                                              ['Wilderness_Area', 'Soil_Type'],
    #                                                              n_jobs=50,
    #                                                              score_type='median')
    # del covtype_vime_classifier
    #
    # print(f"acc_covtype_vime_best: {covtype_vime_best_accuracies}")
    # print(f"acc_covtype_vime_median: {covtype_vime_median_accuracies}")
    # print(f"acc_covtype_vime_worst: {covtype_vime_worst_accuracies}")

    print(f"acc_spotify_tabnet_best: {spotify_tabnet_best_accuracies}")
    # print(f"acc_spotify_tabnet_median: {spotify_tabnet_median_accuracies}")
    # print(f"acc_spotify_tabnet_worst: {spotify_tabnet_worst_accuracies}")

    print(f"acc_spotify_transtab_best: {spotify_transtab_best_accuracies}")
    # print(f"acc_spotify_transtab_median: {spotify_transtab_median_accuracies}")
    # print(f"acc_spotify_transtab_worst: {spotify_transtab_worst_accuracies}")