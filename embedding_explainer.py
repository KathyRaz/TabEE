import warnings
from itertools import product
import time
from tqdm import tqdm
from pathos.pools import ParallelPool

warnings.filterwarnings("ignore")
from pathos.pools import ParallelPool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from constants import DATA_TYPE, DIST, BINS, CATEGORICAL_TYPE, NUMERIC_TYPE, CLUSTER, BINNED_SUFFIX, PROBA_COL
from utils import determine_column_type, equalObs, normalize_by_clusters_num, distributions_distance, \
    distributions_lift, diversity_metric

pd.set_option('mode.chained_assignment', None)


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


class EmbeddingExplainer:
    # TODO assigning an explanation for a single example using the clustering model.
    # TODO exporting Clustering module
    def __init__(self, dataset, embedding, n_clusters_range=range(2, 20, 1), random_state=None, sil_weight=1,
                 suff_weight=1, interest_weight=1, diversity_weight=0.0, explanation_score_type='best'):
        self.dataset = dataset.copy()
        self.embedding = embedding.copy()
        self.source_distributions = dict()
        self.n_clusters_range = list(n_clusters_range)
        self.random_state = random_state if random_state is not None else int(np.random.randint(0, 1e8))
        self.final_clustered_dataset = None
        self.clustering_model = None
        self.n_clusters = None
        self.sil_weight = sil_weight
        self.suff_weight = suff_weight
        self.interest_weight = interest_weight
        self.diversity_weight = diversity_weight
        self.total_weight = self.sil_weight + self.suff_weight + self.interest_weight + self.diversity_weight
        assert self.total_weight > 0.0
        self.explanation_score = None
        self.embedding_explanation = None
        self.umap_representation = None
        self.explanation_score_type = explanation_score_type
        self._bin_dataset_columns()

    def generate_and_plot_explanation(self, candidates_per_cluster=5):
        _ = self.choose_n_clusters_by_explanation(candidates_per_cluster=candidates_per_cluster)
        self.plot_explanation()

    def choose_n_clusters_by_explanation(self, candidates_per_cluster=5):
        candidates_per_cluster = min(candidates_per_cluster, len(self.dataset.columns))
        sil_scores = np.array([])
        cluster_scores = np.array([])
        explanations = []
        for n_clusters in tqdm(self.n_clusters_range, desc='evaluating number of clusters'):
            clustered_dataset, _ = self.cluster_n_clusters(n_clusters, join_dataset_type='both')
            sil_score, mean_clusters_score, explanation = self.evaluate_clustered_dataset(clustered_dataset, n_clusters,
                                                                                          candidates_per_cluster=candidates_per_cluster)
            sil_scores = np.append(sil_scores, sil_score)
            cluster_scores = np.append(cluster_scores, mean_clusters_score)
            explanations.append(explanation)
            del clustered_dataset
        if self.explanation_score_type == 'best':
            argmax_scores = np.argmax(cluster_scores)
        elif self.explanation_score_type == 'median':
            argmax_scores = np.argsort(cluster_scores)[len(cluster_scores) // 2]
        elif self.explanation_score_type == 'worst':
            argmax_scores = np.argsort(cluster_scores)[0]
        best_n = self.n_clusters_range[argmax_scores]
        self.n_clusters = best_n
        self.explanation_score = cluster_scores[argmax_scores]
        self.embedding_explanation = explanations[argmax_scores]
        self.final_clustered_dataset, self.clustering_model = self.cluster_n_clusters(best_n,
                                                                                      join_dataset_type='dataset')
        return self.final_clustered_dataset, self.n_clusters, self.explanation_score, self.embedding_explanation

    def plot_explanation(self):
        assert self.embedding_explanation is not None
        self.plot_clustering_visualization()
        for cluster_id in range(self.n_clusters):
            self._plot_explanation_from_dist(*self.embedding_explanation[cluster_id], cluster_id)
            plt.show()
            # note here that the pattern of explanation must match the input of self.__plot_explanation_from_dist

    def _bin_dataset_columns(self):
        for column in self.dataset.columns:
            self.source_distributions[column] = dict()
            self.source_distributions[column][DATA_TYPE] = determine_column_type(self.dataset[column])
            distribution, bins = bin_single_column(self.dataset[column], type='source',
                                                   data_type=self.source_distributions[column][DATA_TYPE])
            self.source_distributions[column][DIST] = distribution
            self.source_distributions[column][BINS] = bins

    def evaluate_clustered_dataset(self, clustered_dataset, n_clusters, candidates_per_cluster=5):
        # clustered_dataset should be of type original_dataset + embeddings + cluster column
        sil_score = silhouette_score(clustered_dataset[self.embedding.columns], clustered_dataset[CLUSTER])
        clusters_candidates_lists = []
        dataset_columns = list(self.dataset.columns) + [CLUSTER]
        for cluster_id in range(n_clusters):
            cluster_candidates_list = self.cluster_explanation_candidates(cluster_id,
                                                                          num_candidates=candidates_per_cluster,
                                                                          clustered_dataset=clustered_dataset[
                                                                              dataset_columns])
            clusters_candidates_lists.append(cluster_candidates_list)
        all_possible_combinations = list(product(*[list(range(candidates_per_cluster))] * n_clusters))
        num_combinations = len(all_possible_combinations)

        combinations_scores = []
        for combination in all_possible_combinations:
            # combination looks like (0,1,2,0,3,1) with length of n_clusters
            interest_list = []  # interestingness
            sufficiency_list = []
            col_names_list = []
            col_distributions_list = []
            for cluster_id in range(n_clusters):
                cluster_candidate_idx = combination[cluster_id]
                col_name, col_dist, interest, sufficiency = clusters_candidates_lists[cluster_id][cluster_candidate_idx]
                sufficiency_list.append(sufficiency)
                interest_list.append(interest)
                col_names_list.append(col_name)
                col_distributions_list.append(col_dist)
            macro_avg_sufficiency = np.average(sufficiency_list)
            normalized_sufficiency = normalize_by_clusters_num(macro_avg_sufficiency, n_clusters)
            avg_interest = np.average(interest_list)
            if self.diversity_weight > 0.0:
                combination_diversity = diversity_metric(col_names_list, col_distributions_list)
            else:
                combination_diversity = 0.0
            combination_score = 1 / self.total_weight * (self.sil_weight * sil_score +
                                                         self.suff_weight * normalized_sufficiency +
                                                         self.interest_weight * avg_interest +
                                                         self.diversity_weight * combination_diversity)
            combinations_scores.append(combination_score)

        if self.explanation_score_type == 'best':
            argmax_score = np.argmax(combinations_scores)
        elif self.explanation_score_type == 'median':
            argmax_score = np.argsort(combinations_scores)[len(combinations_scores) // 2]
        elif self.explanation_score_type == 'worst':
            argmax_score = np.argsort(combinations_scores)[0]
        max_clusters_score = combinations_scores[argmax_score]
        best_combination = all_possible_combinations[argmax_score]
        best_explanation = [(clusters_candidates_lists[cluster_id][best_combination[cluster_id]][0],
                             clusters_candidates_lists[cluster_id][best_combination[cluster_id]][1]) for cluster_id in
                            range(n_clusters)]
        # 0 index - best column, 1 index - best column's distribution
        return sil_score, max_clusters_score, best_explanation

    def evaluate_cluster_sufficiency(self, clustered_dataset, cluster_id, explained_col, explained_dist):
        explained_lift = distributions_lift(self.source_distributions[explained_col][DIST], explained_dist)
        binned_explained_col = explained_col + BINNED_SUFFIX
        if self.source_distributions[explained_col][DATA_TYPE] == NUMERIC_TYPE:
            bins = self.source_distributions[explained_col][BINS]
            clustered_dataset.loc[:, binned_explained_col] = pd.cut(clustered_dataset[explained_col], bins=bins,
                                                                    labels=bins[:-1], right=False, duplicates='drop')
            clustered_dataset.loc[:, binned_explained_col] = clustered_dataset[binned_explained_col].astype(bins.dtype)
            binned_proba_df = explained_lift.reset_index(name=PROBA_COL).rename(columns={'index': binned_explained_col})
        else:
            clustered_dataset.loc[:, binned_explained_col] = clustered_dataset[explained_col]
            binned_proba_df = explained_lift.reset_index(name=PROBA_COL).rename(
                columns={explained_col: binned_explained_col})
        clustered_dataset = clustered_dataset.merge(binned_proba_df, on=binned_explained_col, how='left')
        full_dataset_proba_sum = clustered_dataset[PROBA_COL].sum()
        cluster_proba_sum = clustered_dataset[clustered_dataset[CLUSTER] == cluster_id][PROBA_COL].sum()
        if cluster_proba_sum == 0:
            return 0.0
        return cluster_proba_sum / full_dataset_proba_sum

    def cluster_n_clusters(self, n_clusters, join_dataset_type='embedding', clustering_method='kmeans'):
        if clustering_method == 'kmeans':
            clustering_model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        elif clustering_method == 'gmm' or clustering_method == 'gussianmixture':
            clustering_model = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        else:
            raise Exception("wrong clustering_method value")
        clusters = clustering_model.fit_predict(self.embedding)
        clustered_dataset = pd.DataFrame(data=clusters, columns=[CLUSTER], index=self.embedding.index)
        if join_dataset_type == 'embedding':
            joined_dataset = self.embedding
        elif join_dataset_type == 'dataset':
            joined_dataset = self.dataset
        elif join_dataset_type == 'both':
            joined_dataset = self.dataset.join(self.embedding)
        else:
            raise Exception("wrong join_dataset_type value")
        clustered_dataset = joined_dataset.join(clustered_dataset)
        return clustered_dataset, clustering_model

    # TODO add to valid latex
    # TODO cut max values in categorical depending on importance
    def _plot_explanation_from_dist(self, column_name, result_dist, cluster_id):
        source_dist = self.source_distributions[column_name][DIST]
        bins = self.source_distributions[column_name][BINS]
        fig, ax = plt.subplots()
        width = 0.35

        if self.source_distributions[column_name][DATA_TYPE] == CATEGORICAL_TYPE:
            label_tags = list(self.source_distributions[column_name][DIST].index)[:20]
            ind = np.arange(len(label_tags))
            ax.bar(ind + width, list(result_dist.loc[label_tags] * 100), width, label=f"Cluster {cluster_id}")
            ax.bar(ind, list(source_dist.loc[label_tags] * 100), width, label="Dataset")
        else:
            ind = np.arange(len(set(source_dist.keys())))
            label_tags = tuple(
                [f"[{bins[i]}, {bins[i + 1]})" if i < len(ind) - 1 else f"[{bins[i]}, {bins[i + 1] - 1}]" for i in ind])
            ax.bar(ind + width, list(result_dist * 100), width, label=f"Cluster {cluster_id}")
            ax.bar(ind, list(source_dist * 100), width, label="Dataset")

        ax.set_xticks(ind + width / 2)
        tags_max_length = max([len(str(tag)) for tag in label_tags])
        ax.set_xticklabels(label_tags, rotation='vertical' if tags_max_length >= 4 else 'horizontal')
        plt.legend(loc='best')
        plt.xlabel(f"{column_name} column values", fontsize=16)
        plt.ylabel("frequency (%)", fontsize=16)

        ax.set_title(
            label=f"Cluster {cluster_id} Explanation\nThe column `{column_name}` is represented significantly differently in the cluster.\n",
            loc='center', wrap=True)
        fig = plt.gcf()
        return fig

    def cluster_explanation_candidates(self, cluster_id, num_candidates=1, clustered_dataset=None):
        # if clustered_dataset is None, the function accesses self.final_clustered_dataset
        assert self.final_clustered_dataset is not None or clustered_dataset is not None
        if clustered_dataset is not None:
            cluster_data = self.dataset[clustered_dataset[CLUSTER] == cluster_id]
        else:
            clustered_dataset = self.final_clustered_dataset
            cluster_data = self.dataset[self.final_clustered_dataset[CLUSTER] == cluster_id]
        columns_distances = []
        columns_distributions = []

        for column in cluster_data.columns:
            column_distribution, _ = bin_single_column(cluster_data[column], type='result',
                                                       data_type=self.source_distributions[column][DATA_TYPE],
                                                       bins=self.source_distributions[column][BINS])
            column_distance, column_distribution = distributions_distance(self.source_distributions[column][DIST],
                                                                          column_distribution,
                                                                          self.source_distributions[column][
                                                                              DATA_TYPE])
            # above, column_distribution is returned again so it contains also empty values that appear in source column
            columns_distances.append(column_distance)
            columns_distributions.append(column_distribution)

        sorted_indices = np.argsort(columns_distances)[::-1]  # note that we sort from highest to lowest
        top_candidates = sorted_indices[:num_candidates]
        top_columns = list(np.array(cluster_data.columns)[top_candidates])
        top_distances = list(np.array(columns_distances)[top_candidates])
        top_distributions = list(np.array(columns_distributions, dtype='object')[top_candidates])

        top_sufficiencies = []
        for candidate_index in range(num_candidates):
            sufficiency = self.evaluate_cluster_sufficiency(clustered_dataset, cluster_id,
                                                            top_columns[candidate_index],
                                                            top_distributions[candidate_index])
            top_sufficiencies.append(sufficiency)
        return list(zip(top_columns, top_distributions, top_distances, top_sufficiencies))

    def plot_clustering_visualization(self, max_clusters=20):
        n_clusters_to_plot = self.n_clusters
        if max_clusters < self.n_clusters:
            n_clusters_to_plot = max_clusters
            print(
                f"Note that only {max_clusters} out of {self.n_clusters} will be shown in the bar.\n"
                f"You can change this behaviour by setting the parameter \"max_clusters\""
            )
        if self.umap_representation is None:
            self.umap_representation = umap.UMAP(random_state=self.random_state).fit_transform(self.embedding)
        plt.scatter(self.umap_representation[:, 0], self.umap_representation[:, 1],
                    c=self.final_clustered_dataset[CLUSTER], cmap="Spectral", s=0.5)
        plt.colorbar(boundaries=np.arange(n_clusters_to_plot + 1) - 0.5, label='cluster',
                     ticks=np.arange(n_clusters_to_plot))
        plt.show()
