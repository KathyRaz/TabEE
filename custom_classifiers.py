import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.abspath('..'))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tabnet_embedding.main as tabnet_embedding
import transtab_main.create_embeddings as transtab_embedding
import torch
from sklearn.linear_model import LogisticRegression


class TabNetCustomClassifier:
    _MODEL_PATH = '/specific/disk1/home/ronycopul/Projects/tabnet_embedding/{name}/tabnet'

    def __init__(self, name, max_epochs=100, not_cat_cols=[]):
        self.name = name
        self.model_path = self._MODEL_PATH.format(name=self.name)
        self.max_epochs = max_epochs
        self.not_cat_cols = not_cat_cols
        self.tabnet_params = None
        self.labels_mapping = None

    def fit(self, X, y):
        X['target_column'] = y
        tabnet_params, X_train, y_train, X_valid, y_valid, labels_mapping = tabnet_embedding.train_preprocess(X,
                                                                                                              'target_column',
                                                                                                              sample_size=len(
                                                                                                                  X),
                                                                                                              not_cat_cols=self.not_cat_cols)
        tabnet_embedding.train(tabnet_params, X_train, y_train, X_valid, y_valid, self.model_path,
                               max_epochs=self.max_epochs)
        self.tabnet_params = tabnet_params
        self.labels_mapping = labels_mapping
        with torch.no_grad():
            X_embedding = tabnet_embedding.get_latent_layer(self.model_path + '.zip',
                                                            np.concatenate([X_train, X_valid]))
            self.classifier = LogisticRegression().fit(X_embedding, np.concatenate([y_train, y_valid]))
        return self

    def predict(self, X):
        X_embedding = self.get_embedding(X)
        predictions = self.classifier.predict(X_embedding)
        return predictions

    def get_embedding(self, X):
        assert self.tabnet_params is not None
        with torch.no_grad():
            X_embedding = pd.DataFrame(
                tabnet_embedding.get_latent_layer(self.model_path + '.zip',
                                                  tabnet_embedding.infer_preprocess(X, self.labels_mapping)),
                index=X.index)
        return X_embedding


class TransTabCustomClassifier:
    def __init__(self, name, num_cols, bin_cols, cat_cols, target_col, num_epoch=30, lr=1e-4):
        self.name, self.num_cols, self.bin_cols, self.cat_cols, self.target_col, self.num_epoch, self.lr = name, num_cols, bin_cols, cat_cols, target_col, num_epoch, lr

    def fit(self, X_fit, y):
        if self.target_col in X_fit:
            X = X_fit.drop(columns=[self.target_col])
        else:
            X = X_fit
        X['target_column'] = y
        transtab_embedding.transtab_train(X, self.name, self.num_cols, self.bin_cols, self.cat_cols, 'target_column',
                                          num_epoch=self.num_epoch, lr=self.lr)
        train_embedding = self.get_embedding(X)
        self.classifier = LogisticRegression().fit(train_embedding, y)
        return self

    def get_embedding(self, X_embed):
        if self.target_col in X_embed:
            X = X_embed.drop(columns=[self.target_col])
        else:
            X = X_embed
        return transtab_embedding(X, self.name, self.num_cols, self.bin_cols, self.cat_cols, 'target_column')

    def predict(self, X):
        X_embedding = self.get_embedding(X)
        predictions = self.classifier.predict(X_embedding)
        return predictions
