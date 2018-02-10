"""Preprocess raw data
"""
import gc
import pathlib

import pandas as pd
import numpy as np
from joblib import Memory
from sklearn import preprocessing

MEMORY = Memory(cachedir="cache/", verbose=1)


@MEMORY.cache
def read_data():
    print("Reading data...")
    pathlib.Path("cache/").mkdir(parents=True, exist_ok=True)
    df_train = pd.read_csv(
        'data/train.csv', usecols=[1, 2, 3, 4, 5],
        dtype={
            'store_nbr': np.int32, "item_nbr": np.int32
        },
        converters={'unit_sales': lambda u: np.log1p(
            float(u)) if u != "" and float(u) > 0 else 0},
        engine="c",
        parse_dates=["date"],
        skiprows=range(1, 16322662)  # 2014-01-01
    )

    df_test = pd.read_csv(
        "data/test.csv", usecols=[0, 1, 2, 3, 4],
        parse_dates=["date"]
    ).set_index(
        ['store_nbr', 'item_nbr', 'date']
    )

    stores = pd.read_csv(
        "data/stores.csv",
        converters={"type": lambda x: ord(x) - ord("A")},
    ).set_index("store_nbr")
    cluster_dict = stores["cluster"].to_dict()

    items = pd.read_csv(
        "data/items.csv",
    ).set_index("item_nbr")
    items["family"] = preprocessing.LabelEncoder(
    ).fit_transform(items["family"])
    items["class"] = preprocessing.LabelEncoder().fit_transform(items["class"])

    df_train = pd.concat([
        df_train[df_train.date.isin(
            pd.date_range("2014-03-01", "2014-09-30"))],
        df_train[df_train.date.isin(
            pd.date_range("2015-03-01", "2015-09-30"))],
        df_train[df_train.date.isin(
            pd.date_range("2016-03-01", "2016-09-30"))],
        df_train[df_train.date.isin(
            pd.date_range("2017-01-01", "2017-08-15"))]
    ], axis=0)
    gc.collect()

    df_train = df_train.merge(
        items[["family", "class"]], left_on="item_nbr", right_index=True
    ).merge(
        stores[["cluster", "type"]], left_on="store_nbr", right_index=True
    )

    # Promotion
    print("Calculating df_promo...")
    df_promo = df_train.set_index(
        ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
            level=-1).fillna(False)
    df_promo.columns = df_promo.columns.get_level_values(1)
    df_promo_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    df_promo_test.columns = df_promo_test.columns.get_level_values(1)
    df_promo_test = df_promo_test.reindex(df_promo.index).fillna(False)
    df_promo = pd.concat([df_promo, df_promo_test], axis=1)
    del df_promo_test

    # Sales numbers for each store
    print("Calculating df_sales...")
    df_sales = df_train.set_index(
        ["store_nbr", "item_nbr", "date"])["unit_sales"].unstack(
            level=-1).fillna(0)

    # The date of the first sales of an item in a store
    print("Calculating df_first_date...")
    df_first_date = df_train.groupby(
        ["store_nbr", "item_nbr"])["date"].min().reindex(
            df_sales.index, fill_value=pd.to_datetime("2017-08-31"))

    # Sales numbers for each cluster
    print("Calculating df_sales_cluster...")
    df_sales_cluster = df_train.groupby(
        ["cluster", "item_nbr", "date"])["unit_sales"].sum(
    ).unstack(level=-1).fillna(0)
    df_sales_cluster = df_sales_cluster.reindex([
        (cluster_dict[x[0]], x[1]) for x in df_sales.index
    ])

    # Mean item class sales in the same store
    print("Calculating df_item_class...")
    df_class_means = df_train.groupby(["date", "store_nbr", "class"])[
        "unit_sales"].mean().to_frame("class_mean")
    df_class_means = df_train[["date", "store_nbr", "item_nbr", "class"]].merge(
        df_class_means, left_on=["date", "store_nbr", "class"], right_index=True
    ).reset_index()
    df_class_means = df_class_means.set_index(
        ["store_nbr", "item_nbr", "date"])["class_mean"].unstack(
            level=-1).fillna(0)

    del df_train
    gc.collect()

    print("Reindexing stores and items")
    # Store clusters, types, and cities
    stores = stores[["cluster", "type", "city"]].reindex(
        df_sales.index.get_level_values(0))

    # Item families and classes
    items = items.reindex(df_sales.index.get_level_values(1))

    return (
        df_sales,
        df_promo,
        stores,
        items,
        df_class_means,
        df_sales_cluster,
        df_first_date
    )
