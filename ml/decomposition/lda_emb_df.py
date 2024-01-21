import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from lda_emb import LDAEmb


class LDAEmbDf:
    def __init__(
        self,
        cat_cols=[],
        emb_col_format="{}_{}_e{:03d}",
        **kwargs,
    ):

        self.cat_cols = cat_cols
        self._ldae = {}

        for col_x in self.cat_cols:
            for col_y in self.cat_cols:
                if col_x == col_y:
                    continue

                ldae = LDAEmb(
                    col_x=col_x, col_y=col_y, emb_col_format=emb_col_format, **kwargs
                )
                self._ldae[(col_x, col_y)] = ldae

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit(self, df):

        for col_x in self.cat_cols:
            for col_y in self.cat_cols:
                if col_x == col_y:
                    continue

                x = df[col_x]
                y = df[col_y]

                self._ldae[(col_x, col_y)].fit(x, y)

    def transform(self, df):

        transformed_list = []
        for col_x in self.cat_cols:
            for col_y in self.cat_cols:
                if col_x == col_y:
                    continue

                x = df[col_x]

                transformed_df = self._ldae[(col_x, col_y)].transform(x)
                transformed_list.append(transformed_df)
        original_df = df.drop(self.cat_cols)
        return pl.concat([original_df] + transformed_list, how="horizontal")


if __name__ == "__main__":
    num_rows = 100
    num_cats = 5
    n_components = 3

    rng = np.random.RandomState(0)

    cat_cols = ["col_1", "col_2", "col_3"]
    df_dict = {
        cat_col: rng.randint(num_cats, size=num_rows).astype(str)
        for cat_col in cat_cols
    }
    df_dict["index"] = np.arange(num_rows)
    df = pl.DataFrame(df_dict)
    ldaed = LDAEmbDf(
        cat_cols=cat_cols,
        n_components=n_components,
        random_state=rng,
    )

    fit_transformed_df = ldaed.fit_transform(df)
    print(fit_transformed_df)

    transformed_df = ldaed.transform(df)
    assert_frame_equal(fit_transformed_df, transformed_df)
