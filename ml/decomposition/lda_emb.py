import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from sklearn.decomposition import LatentDirichletAllocation


class LDAEmb:
    def __init__(
        self,
        col_1,
        col_2,
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

    def fit(self, df):
        self.cooccurence_df, self.cooccurence_2darr = self.to_cooccurence_matrix(df)
        self._lda.fit(self.cooccurence_2darr)

    def transform(self, df):
        cooccurence_df, cooccurence_2darr = self.to_cooccurence_matrix(df)
        return self.transform_cooccurence(cooccurence_df, cooccurence_2darr)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform_cooccurence(self.cooccurence_df, self.cooccurence_2darr)

    def transform_cooccurence(self, cooccurence_df, cooccurence_2darr):
        col_1 = self.col_1
        col_2 = self.col_2
        emb_col_format = self.emb_col_format

        emb_2darr = self._lda.transform(cooccurence_2darr)

        emb_col_list = [
            emb_col_format.format(col_1, col_2, i) for i in range(emb_2darr.shape[1])
        ]
        emb_df = pl.DataFrame(emb_2darr, schema=emb_col_list)
        emb_df = emb_df.with_columns([cooccurence_df[col_1]])
        out_df = df.join(
            emb_df,
            on=col_1,
            how="left",
        )
        out_df = out_df.drop([col_1, col_2])
        return out_df


if __name__ == "__main__":

    num_rows = 100
    num_cats_1 = 7
    num_cats_2 = 5
    n_components = 3
    col_1 = "col_1"
    col_2 = "col_2"

    rng = np.random.RandomState(0)
    df_dict = {
        "index": np.arange(num_rows),
        col_1: rng.randint(num_cats_1, size=num_rows),
        col_2: rng.randint(num_cats_2, size=num_rows),
    }

    df = pl.DataFrame(df_dict)

    ldae = LDAEmb(
        col_1=col_1,
        col_2=col_2,
        n_components=n_components,
        learning_method="batch",
        random_state=rng,
    )

    fit_transformed_df = ldae.fit_transform(df)
    assert fit_transformed_df.width == df.width - 2 + n_components
    assert len(fit_transformed_df) == len(df)
    print(df)
    print(fit_transformed_df)

    transformed_df = ldae.transform(df)
    assert transformed_df.width == df.width - 2 + n_components
    assert len(transformed_df) == len(df)

    assert_frame_equal(transformed_df, fit_transformed_df)
