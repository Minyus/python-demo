import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

if __name__ == "__main__":
    """
    Reference:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/tests/test_online_lda.py#L133
    """
    rng = np.random.RandomState(0)
    x_2darr = rng.randint(10, size=(7, 5))
    assert x_2darr.shape == (7, 5)
    lda = LatentDirichletAllocation(
        n_components=3, learning_method="batch", random_state=rng
    )
    lda.fit(x_2darr)
    xt_2darr = lda.transform(x_2darr)
    assert xt_2darr.shape == (7, 3)
