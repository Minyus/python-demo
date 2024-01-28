import numpy as np
import pandas as pd
import polars as pl
import polars.selectors as cs
from sklearn.decomposition import LatentDirichletAllocation


class LDAEmb:
    def __init__(
        self,
        col_x="col_x",
        col_y="col_y",
        emb_col_format="{}_{}_e{:03d}",
        minimize_sort=False,
        emb_dtype="float32",
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
        self.emb_dtype = emb_dtype
        self._lda = LatentDirichletAllocation(**kwargs)

    def to_cooccurence_matrix(self, df):
        col_x = self.col_x
        col_y = self.col_y
        minimize_sort = self.minimize_sort

        col_list = [col_x, col_y]

        """
        "count" renamed to "len" in polars 0.20.5
        https://github.com/pola-rs/polars/releases/tag/py-0.20.5
        """
        _agg = "count" if hasattr(df.group_by(col_list), "count") else "len"

        count_df = getattr(df.group_by(col_list), _agg)()
        count_df = count_df.sort(by=[col_x] if minimize_sort else col_list)
        cooccurence_df = count_df.pivot(
            index=col_x, columns=col_y, values=_agg
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
        if self.emb_dtype is not None:
            assert isinstance(self.emb_dtype, str)
            assert hasattr(np, self.emb_dtype)
            emb_2darr = emb_2darr.astype(getattr(np, self.emb_dtype))
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
        assert len(out_df) == len(df), f"{len(out_df)} != {len(df)}"
        out_df = out_df.drop([self.col_x])
        if isinstance(x, np.ndarray):
            return out_df.to_numpy()
        return out_df


class LDAEmbDf:
    def __init__(
        self,
        cat_cols=None,
        keep_original=False,
        verbose=True,
        **kwargs,
    ):

        self.cat_cols = cat_cols
        self.keep_original = keep_original
        self.verbose = verbose
        self.kwargs = kwargs

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def to_polars(self, df):
        self.pandas_input = False

        if not isinstance(df, pl.DataFrame):
            df = df.copy(deep=True)
            pd_categorical_cols = df.select_dtypes(include="category").columns.tolist()
            self.pd_categories_dict = {}
            for col in pd_categorical_cols:
                self.pd_categories_dict[col] = df[col].cat.categories
                df[col] = df[col].astype(str)
            df = pl.from_pandas(df)
            self.pandas_input = True
            if self.verbose:
                print("Input Pandas DataFrame")
        return df

    def fit(self, df):

        df = self.to_polars(df)

        if self.cat_cols is None:
            self.cat_cols = list(df.select(cs.string(include_categorical=True)).columns)

        assert len(self.cat_cols) >= 2, (
            "Specify 2 or more categorical columns in: " + df.schema
        )
        self._ldae = {}

        for col_x in self.cat_cols:
            for col_y in self.cat_cols:
                if col_x == col_y:
                    continue

                ldae = LDAEmb(col_x=col_x, col_y=col_y, **self.kwargs)
                self._ldae[(col_x, col_y)] = ldae

                x = df[col_x]
                y = df[col_y]

                self._ldae[(col_x, col_y)].fit(x, y)
                if self.verbose:
                    print(f"LDA model fit on {col_x} with {col_y}")

    def transform(self, df):
        num_rows = len(df)
        if isinstance(df, pl.DataFrame):
            cat_df = df.select(self.cat_cols)
            keep_df = df.drop(self.cat_cols)
        else:
            cat_df_pd = df[self.cat_cols]
            cat_df = self.to_polars(cat_df_pd)
            keep_df_pd = df.drop(self.cat_cols, axis=1)

        transformed_list = []
        for col_x in self.cat_cols:
            for col_y in self.cat_cols:
                if col_x == col_y:
                    continue

                x = cat_df[col_x]

                transformed_df = self._ldae[(col_x, col_y)].transform(x)
                assert (
                    len(transformed_df) == num_rows
                ), f"{len(transformed_df)} != {num_rows}"
                if self.verbose:
                    print(
                        f"From {col_x}, LDA model fit with {col_y} generated: {transformed_df.columns}"
                    )
                transformed_list.append(transformed_df)

        transformed_df = pl.concat(transformed_list, how="horizontal")
        assert len(transformed_df) == num_rows, f"{len(transformed_df)} != {num_rows}"

        if isinstance(df, pl.DataFrame):
            if self.keep_original:
                df_list = [df, transformed_df]
            else:
                df_list = [keep_df, transformed_df]
            out_df = pl.concat(df_list, how="horizontal")
            assert len(out_df) == num_rows, f"{len(out_df)} != {num_rows}"
            return out_df
        else:
            transformed_df_pd = transformed_df.to_pandas().reset_index(drop=True)
            if self.keep_original:
                pd_df_list = [df.reset_index(drop=True), transformed_df_pd]
            else:
                pd_df_list = [keep_df_pd.reset_index(drop=True), transformed_df_pd]
            out_df = pd.concat(pd_df_list, axis=1)
            assert len(out_df) == num_rows, f"{len(out_df)} != {num_rows}"
            return out_df


if __name__ == "__main__":

    def test_ldaemb():
        """Test LDAEmb"""
        from numpy.testing import assert_array_almost_equal

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

    def test_ldaembdf():
        """Test LDAEmbDf"""
        from polars.testing import assert_frame_equal
        from pandas.testing import assert_frame_equal as pd_assert_frame_equal

        num_rows = 10
        num_cats = 5
        n_components = 3

        rng = np.random.RandomState(0)

        cat_cols = ["col_1", "col_2"]
        df_dict = {
            cat_col: rng.randint(num_cats, size=num_rows).astype(str)
            for cat_col in cat_cols
        }
        df_dict["index"] = np.arange(num_rows)
        df_dict["test_flag"] = df_dict["index"] < 3

        for df in [pl.DataFrame(df_dict), pd.DataFrame(df_dict)]:
            ldaed = LDAEmbDf(
                cat_cols=None,
                n_components=n_components,
                random_state=rng,
            )

            fit_transformed_df = ldaed.fit_transform(df)
            # print(fit_transformed_df)

            transformed_df = ldaed.transform(df)
            assert type(fit_transformed_df) == type(transformed_df), (
                type(fit_transformed_df).__str__() + "\n!=\n" + type(transformed_df)
            )

            other_cols = [col for col in df.columns if col not in cat_cols]
            if isinstance(df, pl.DataFrame):
                assert_frame_equal(fit_transformed_df, transformed_df)
                assert_frame_equal(
                    fit_transformed_df.select(other_cols), df.select(other_cols)
                )
            else:
                pd_assert_frame_equal(fit_transformed_df, transformed_df)
                pd_assert_frame_equal(
                    fit_transformed_df[other_cols], transformed_df[other_cols]
                )

    test_ldaemb()
    test_ldaembdf()
