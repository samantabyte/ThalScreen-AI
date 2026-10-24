
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


class CorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)

        corr_matrix = X_df.corr(method="pearson").abs()
        upper_mask = np.triu(
            np.ones(corr_matrix.shape, dtype=bool), k=1
        )
        upper_triangle = corr_matrix.where(upper_mask)

        self.features_to_drop_ = [
            column for column in upper_triangle.columns
            if (upper_triangle[column] >= self.threshold).any()
        ]

        self.features_to_keep_ = [
            column for column in X_df.columns
            if column not in self.features_to_drop_
        ]

        self.n_features_in_ = X_df.shape[1]
        self.n_features_out_ = len(self.features_to_keep_)
        return self

    def transform(self, X):
        X_df = (
            X.copy() if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=self.feature_names_in_)
        )
        return X_df[self.features_to_keep_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.features_to_keep_, dtype=object)


class AdaptiveMISelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        self.effective_k_ = min(self.k, X_df.shape[1])

        self.mi_scores_ = mutual_info_classif(
            X_df, y, random_state=42
        )

        ranked_indices = np.argsort(self.mi_scores_)[::-1]
        self.selected_indices_ = ranked_indices[:self.effective_k_]
        self.selected_features_ = self.feature_names_in_[
            self.selected_indices_
        ].tolist()
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_].copy()
        return np.asarray(X)[:, self.selected_indices_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_features_, dtype=object)
