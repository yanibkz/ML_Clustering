# custom_clustering.py
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
import numpy as np

class ClusterWithSplit(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, cluster_to_split=3, sub_n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.cluster_to_split = cluster_to_split
        self.sub_n_clusters = sub_n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_main = KMeans(n_clusters=self.n_clusters, n_init=12, random_state=self.random_state)
        self.labels_ = self.kmeans_main.fit_predict(X)
        mask = self.labels_ == self.cluster_to_split
        self.idx_split = np.where(mask)[0]
        X_split = X[self.idx_split]
        self.kmeans_sub = KMeans(n_clusters=self.sub_n_clusters, n_init=12, random_state=self.random_state)
        self.sub_labels_ = self.kmeans_sub.fit_predict(X_split)
        self.labels_final_ = self.labels_.astype(object).astype(str)
        for s in range(self.sub_n_clusters):
            self.labels_final_[self.idx_split[self.sub_labels_ == s]] = f"{self.cluster_to_split}_{s+1}"
        return self

    def predict(self, X):
        main_labels = self.kmeans_main.predict(X)
        final_labels = main_labels.astype(object).astype(str)
        mask = main_labels == self.cluster_to_split
        X_split = X[mask]
        if len(X_split) > 0:
            sub_labels = self.kmeans_sub.predict(X_split)
            idx = np.where(mask)[0]
            for s in range(self.sub_n_clusters):
                final_labels[idx[sub_labels == s]] = f"{self.cluster_to_split}_{s+1}"
        return final_labels
