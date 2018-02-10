from datetime import date, timedelta
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import joblib

VAL_IDX_FILEPATH = "cache/val_idx.pkl"
TEST_IDX_FILEPATH = "cache/test_idx.pkl"

VAL_START_DATE = date(2017, 7, 26)
TEST_START_DATE = date(2017, 8, 16)


def transform_predictions(pred, idx_filepath, start_date):
    idx_store_item = joblib.load(idx_filepath)
    df_preds = pd.DataFrame(
        pred, index=idx_store_item,
        columns=pd.date_range(start_date, periods=16)
    ).stack().to_frame("unit_sales")
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
    df_preds["unit_sales"] = np.clip(np.expm1(df_preds.unit_sales), 0, 1e5)
    return df_preds


def create_base_dir(filepath):
    base_dir = PurePosixPath(filepath).parent
    Path(base_dir).mkdir(parents=True, exist_ok=True)


def export_validation(filepath, val_pred):
    create_base_dir(filepath)
    df_preds = transform_predictions(
        val_pred, VAL_IDX_FILEPATH, VAL_START_DATE)
    df_preds.to_csv(filepath, float_format="%.6f", index=True)


def export_test(filepath, test_pred):
    create_base_dir(filepath)
    df_test = pd.read_csv(
        "data/test.csv", usecols=[0, 1, 2, 3, 4],
        dtype={'onpromotion': bool},
        parse_dates=["date"]
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )
    df_preds = transform_predictions(
        test_pred, TEST_IDX_FILEPATH, TEST_START_DATE)
    df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)
    df_preds = df_test[["id"]].join(df_preds, how="left").fillna(0)
    df_preds.to_csv(filepath, float_format="%.6f", index=False)
