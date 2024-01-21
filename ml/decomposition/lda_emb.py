import numpy as np
from numpy.testing import assert_array_almost_equal
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation


class LDAEmb:
    def __init__(
        self,
        col_x="col_x",
        col_y="col_y",
        emb_col_format="{}_{}_e{:03d}",
        minimize_sort=False,
        **kwargs,
    ):
        """
        Reference:
        https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/tests/test_online_lda.py#L133
        """
        self.col_x = col_x
        self.col_y = col_y
        self.emb_col_format = emb_col_format
        self.minimize_sort = minimize_sort
        self._lda = LatentDirichletAllocation(**kwargs)

    def to_cooccurence_matrix(self, df):
        col_x = self.col_x
        col_y = self.col_y
        minimize_sort = self.minimize_sort

        col_list = [col_x, col_y]
        count_df = (
            df.group_by(col_list).len().sort(by=[col_x] if minimize_sort else col_list)
        )
        cooccurence_df = count_df.pivot(
            index=col_x, columns=col_y, values="len"
        ).fill_null(0)
        cooccurence_2darr = cooccurence_df.drop(col_x).to_numpy()
        return cooccurence_df, cooccurence_2darr

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x, y):
        assert x.shape == y.shape
        df = pl.DataFrame(
            {
                self.col_x: x,
                self.col_y: y,
            }
        )
        cooccurence_df, cooccurence_2darr = self.to_cooccurence_matrix(df)
        emb_2darr = self._lda.fit_transform(cooccurence_2darr)
        emb_col_list = [
            self.emb_col_format.format(self.col_x, self.col_y, i)
            for i in range(emb_2darr.shape[1])
        ]
        emb_df = pl.DataFrame(emb_2darr, schema=emb_col_list)
        self._emb_df = emb_df.with_columns([cooccurence_df[self.col_x]])

    def transform(self, x):

        df = pl.DataFrame({self.col_x: x})

        out_df = df.join(
            self._emb_df,
            on=self.col_x,
            how="left",
        )
        out_df = out_df.drop([self.col_x])
        if isinstance(x, np.ndarray):
            return out_df.to_numpy()
        return out_df


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
        random_state=rng,
    )

    fit_transformed_2darr = ldae.fit_transform(x, y)
    assert fit_transformed_2darr.shape == (x.shape[0], n_components)

    transformed_2darr = ldae.transform(x)
    assert transformed_2darr.shape == (x.shape[0], n_components)

    assert_array_almost_equal(fit_transformed_2darr, transformed_2darr)

    assert_array_almost_equal(
        ldae.transform(np.array([-1])), np.array([[np.NaN] * n_components])
    )
