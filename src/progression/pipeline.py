"""Feature pipeline shared by every progression experiment (decision 33).

Kept in its own module so saved models (joblib) can be loaded by any script, not only by the
one that trained them.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class Features:
    """Standardise voxels -> PCA (k_max) -> standardise the first k scores [+ standardised age].

    Fitted on training rows only. PCA is fitted once with k_max components and the first k are
    used, so every k in the grid shares one decomposition (as in the brain-age baseline)."""

    def __init__(self, k_max: int, use_age: bool, seed: int):
        self.k_max, self.use_age, self.seed = k_max, use_age, seed

    def fit(self, X, age):
        self.vox = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.k_max, svd_solver="randomized",
                       random_state=self.seed).fit(self.vox.transform(X))
        scores = self.pca.transform(self.vox.transform(X))
        self.score_scale = StandardScaler().fit(scores)
        self.age_scale = StandardScaler().fit(age)
        return self

    def transform(self, X, age, k: int):
        scores = self.score_scale.transform(self.pca.transform(self.vox.transform(X)))[:, :k]
        if self.use_age:
            return np.hstack([scores, self.age_scale.transform(age)])
        return scores
