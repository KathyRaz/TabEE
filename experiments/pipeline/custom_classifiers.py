import numpy as np
import pandas as pd
import sys, os

from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, os.path.abspath('../../..'))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tabnet_embedding.main as tabnet_embedding
import transtab_main.create_embeddings as transtab_embedding
from VIME.vime_self import vime_self
import torch
from sklearn.linear_model import LogisticRegression


class VIMECustomClassifier:

    def __init__(self, name, num_cols, cat_cols, target_col, num_epochs=100, hidden_dim=16):
        self.name = name
        self.num_epochs = num_epochs
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.hidden_dim = hidden_dim
        self.scaler = None
        self.ohe_encoder = None
        self.encoder = None
        self.classifier = None

    def fit(self, X_fit, y):
        X = self._preprocess(X_fit)
        params = {'batch_size': 1024,
                  'epochs': self.num_epochs}
        self.encoder = vime_self(X, p_m=0.3, alpha=2.0, parameters=params, hidden_dim=self.hidden_dim)
        train_embedding = self.get_embedding(X)
        self.classifier = LogisticRegression().fit(train_embedding, y)
        return self

    def _preprocess(self, X):
        X_post = X.copy()
        if self.target_col in X_post.columns:
            X_post = X_post.drop(columns=[self.target_col])
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_post.loc[:, self.num_cols] = self.scaler.fit_transform(X_post[self.num_cols])
        else:
            X_post.loc[:, self.num_cols] = self.scaler.transform(X_post[self.num_cols])
        return X_post

    def get_embedding(self, X_embed):
        assert self.encoder is not None
        X = self._preprocess(X_embed)
        return self.encoder.predict(X)

    def predict(self, X):
        X_embedding = self.get_embedding(X)
        predictions = self.classifier.predict(X_embedding)
        return predictions


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
        X.drop(columns=['target_column'], inplace=True)
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

    def predict_proba(self, X):
        X_embedding = self.get_embedding(X)
        predictions = self.classifier.predict_proba(X_embedding)
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
    def __init__(self, name, num_cols, bin_cols, cat_cols, target_col, hidden_dim=16, num_epoch=30, lr=1e-4):
        self.name = name
        self.num_cols = num_cols
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.num_epoch = num_epoch
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.encoder = None

    def fit(self, X_fit, y):
        if self.target_col in X_fit:
            X = X_fit.drop(columns=[self.target_col])
        else:
            X = X_fit
        X['target_column'] = y
        transtab_embedding.transtab_train(X, self.name, self.num_cols, self.bin_cols, self.cat_cols, 'target_column',
                                          hidden_dim=self.hidden_dim, num_epoch=self.num_epoch, lr=self.lr)
        X.drop(columns=['target_column'], inplace=True)
        train_embedding, self.encoder = self.get_embedding(X)

        self.classifier = LogisticRegression().fit(train_embedding, y)
        return self

    def get_embedding(self, X_embed):
        if self.target_col in X_embed:
            X = X_embed.drop(columns=[self.target_col])
        else:
            X = X_embed
        if self.encoder is None:
            embeddings, encoder = transtab_embedding.transtab_embedding(X, self.name, self.num_cols, self.bin_cols,
                                                                        self.cat_cols, 'target_column',
                                                                        hidden_dim=self.hidden_dim)
            return embeddings, encoder
        else:
            embeddings, _ = transtab_embedding.transtab_embedding(X, self.name, self.num_cols, self.bin_cols,
                                                                  self.cat_cols, 'target_column', enc=self.encoder)
            return embeddings

    def predict(self, X):
        X_embedding = self.get_embedding(X)
        predictions = self.classifier.predict(X_embedding)
        return predictions
