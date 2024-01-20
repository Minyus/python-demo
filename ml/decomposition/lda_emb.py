import numpy as np
from numpy.testing import assert_array_almost_equal
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation


class LDAEmb:
    def __init__(
        self,
        col_1="col_1",
        col_2="col_2",
        emb_col_format="{}_{}_e{:03d}",
        minimize_sort=False,
        **kwargs,
    ):
        """
        Reference:
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/tests/test_online_lda.py#L133
        """
        self.col_1 = col_1
        self.col_2 = col_2
        self.emb_col_format = emb_col_format
        self.minimize_sort = minimize_sort
        self._lda = LatentDirichletAllocation(**kwargs)

    def to_cooccurence_matrix(self, df):
        col_1 = self.col_1
        col_2 = self.col_2
        minimize_sort = self.minimize_sort

        col_list = [col_1, col_2]
        count_df = (
            df.group_by(col_list).len().sort(by=[col_1] if minimize_sort else col_list)
        )
        cooccurence_df = count_df.pivot(
            index=col_1, columns=col_2, values="len"
        ).fill_null(0)
        cooccurence_2darr = cooccurence_df.drop(col_1).to_numpy()
        return cooccurence_df, cooccurence_2darr

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x, y):
        assert x.shape == y.shape
        df = pl.DataFrame(
            {
                self.col_1: x,
                self.col_2: y,
            }
        )
        self.cooccurence_df, self.cooccurence_2darr = self.to_cooccurence_matrix(df)
        self.emb_2darr = self._lda.fit_transform(self.cooccurence_2darr)
        emb_col_list = [
            self.emb_col_format.format(self.col_1, self.col_2, i)
            for i in range(self.emb_2darr.shape[1])
        ]
        emb_df = pl.DataFrame(self.emb_2darr, schema=emb_col_list)
        self.emb_df = emb_df.with_columns([self.cooccurence_df[self.col_1]])

    def transform(self, x):

        df = pl.DataFrame({self.col_1: x})

        out_df = df.join(
            self.emb_df,
            on=self.col_1,
            how="left",
        )
        out_df = out_df.drop([self.col_1])
        return out_df.to_numpy()


if __name__ == "__main__":

    num_rows = 100
    num_cats_1 = 7
    num_cats_2 = 5
    n_components = 3

    rng = np.random.RandomState(0)
    x = rng.randint(num_cats_1, size=num_rows)
    y = rng.randint(num_cats_2, size=num_rows)

    ldae = LDAEmb(
        n_components=n_components,
        learning_method="batch",
        random_state=rng,
    )

    fit_transformed_2darr = ldae.fit_transform(x, y)
    assert fit_transformed_2darr.shape == (x.shape[0], n_components)

    transformed_2darr = ldae.transform(x)
    assert transformed_2darr.shape == (x.shape[0], n_components)

    assert_array_almost_equal(fit_transformed_2darr, transformed_2darr)
