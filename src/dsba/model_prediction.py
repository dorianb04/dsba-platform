import logging
import pandas as pd
from sklearn.base import ClassifierMixin
from dsba.preprocessing import preprocess_dataframe


def classify_dataframe(
    model: ClassifierMixin, df: pd.DataFrame, target_column: str
) -> pd.DataFrame:
    _check_target_column(df, target_column)
    df = preprocess_dataframe(df)
    y_predicted = model.predict(df)
    df[target_column] = y_predicted
    return df


def classify_record(
    model: ClassifierMixin, record: dict, target_column: str
) -> int | float | str:
    df = pd.DataFrame([record])
    _check_target_column(df, target_column)
    df = classify_dataframe(model, df, target_column)
    return df.iloc[0][target_column]


def _check_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    As a convenience, we allow the user to pass a dataframe that already has the target column in the input
    but this is quite suspicious, so we warn the user.
    then we need to drop the target column before we can continue to have the right shape for the prediction
    """
    if target_column in df.columns:
        logging.warning(
            f"Target column {target_column} already exists in the DataFrame."
        )
        df.drop(columns=[target_column], inplace=True)