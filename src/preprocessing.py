# import pandas as pd


# def fit_age_imputer(
#     train_df: pd.DataFrame,
# ) -> tuple[pd.Series, float]:
#     title_age_medians = (
#         train_df
#         .groupby(
#             "Title",
#             observed=True,
#         )["Age"]
#         .median()
#     )

#     global_age_median = train_df["Age"].median()

#     return (
#         title_age_medians,
#         global_age_median,
#     )


# def impute_age(
#     df: pd.DataFrame,
#     title_age_medians: pd.Series,
#     global_age_median: float,
# ) -> pd.DataFrame:
#     result = df.copy()

#     estimated_age = result["Title"].map(
#         title_age_medians
#     )

#     result["Age"] = (
#         result["Age"]
#         .fillna(estimated_age)
#         #안전장치
#         .fillna(global_age_median)
#     )

#     return result

from sklearn.base import BaseEstimator, TransformerMixin

class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.title_age_medians_ = None
        self.global_age_median_ = None

    def fit(self, X, y=None):
        self.title_age_medians_ = (
            X.groupby('Title', observed = True)['Age']
            .median()
        )

        self.global_age_median_ = X['Age'].median()

        return self

    def transform(self, X):
        X = X.copy()

        estimated_age = X['Title'].map(
            self.title_age_medians_
        )

        X['Age'] = (
            X['Age']
            .fillna(estimated_age)
            .fillna(self.global_age_median_)
        )

        return X