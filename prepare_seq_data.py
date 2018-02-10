"""Prepare sequence dataset
"""
from datetime import date, timedelta

import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

from preprocess import read_data
LOOKBACK = 56


def main():
    (
        df_sales, df_promo, stores, items,
        df_class_means, df_sales_cluster, df_first_date
    ) = read_data()

    # Global promotions
    df_promo_global = df_promo.reset_index().drop(
        "store_nbr", axis=1).groupby("item_nbr").sum() / 54
    df_promo_global = df_promo_global.reindex(
        df_sales.index.get_level_values(1), fill_value=0)

    items = items.reset_index(drop=False)
    items["item_nbr"] = preprocessing.LabelEncoder(
    ).fit_transform(items["item_nbr"])
    stores = stores.reset_index(drop=False)
    stores["store_nbr"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["store_nbr"])
    stores["city"] = preprocessing.LabelEncoder(
    ).fit_transform(stores["city"])

    def get_timespan(df, dd, delta, periods):
        return df[
            pd.date_range(dd - timedelta(days=delta), periods=periods)
        ].copy()

    def prepare_dataset(ty1, ty2, is_train=True):
        assert ty1.weekday() == 2
        assert ty2.weekday() == 2
        sales_year1 = get_timespan(df_sales, ty1, LOOKBACK - 1, LOOKBACK + 15)
        sales_cluster_year1 = get_timespan(
            df_sales_cluster, ty1, LOOKBACK - 1, LOOKBACK + 15)
        class_means_year1 = get_timespan(
            df_class_means, ty1, LOOKBACK - 1, LOOKBACK + 15)
        promo_year1 = get_timespan(df_promo, ty1, LOOKBACK - 1, LOOKBACK + 15)
        promo_year2 = get_timespan(df_promo, ty2, LOOKBACK - 1, LOOKBACK + 15)
        promo_global_year1 = get_timespan(
            df_promo_global, ty1, LOOKBACK - 1, LOOKBACK + 15)
        promo_global_year2 = get_timespan(
            df_promo_global, ty2, LOOKBACK - 1, LOOKBACK + 15)
        assert promo_year1.columns.tolist()[0].year == ty1.year
        assert promo_year2.columns.tolist()[0].year == ty2.year
        # Those which include future days:
        sales_year2 = df_sales.reindex(
            columns=[
                x - timedelta(days=1)
                for x in promo_year2.columns
            ],
            fill_value=0
        )
        sales_cluster_year2 = df_sales_cluster.reindex(
            columns=sales_year2.columns, fill_value=0)
        class_means_year2 = df_class_means.reindex(
            columns=sales_year2.columns, fill_value=0)

        # 56-day nonzero filter
        nonzero = (
            (sales_year2.iloc[:, -71:-15].sum(axis=1).values > 0)
        )
        x = np.concatenate(
            [
                np.expand_dims(df.values[nonzero, :], 2)
                for df in (
                    sales_year2, sales_year1,
                    sales_cluster_year2, sales_cluster_year1,
                    class_means_year2, class_means_year1,
                )
            ], axis=2
        ).astype("float64")
        x_int = np.concatenate(
            [
                np.expand_dims(df.values[nonzero, :], 2)
                for df in (promo_year2, promo_year1,
                           promo_global_year2, promo_global_year1)
            ] +
            [
                np.repeat(
                    items["class"].values[nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    items["family"].values[nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    stores["type"].values[nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    stores["cluster"].values[
                        nonzero, np.newaxis, np.newaxis] - 1,
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    stores["store_nbr"].values[
                        nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    stores["city"].values[
                        nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                # DAY
                np.repeat(
                    sales_year2.columns.day.values[
                        np.newaxis, :, np.newaxis],
                    sum(nonzero), axis=0) - 1,
                # MONTH
                np.repeat(
                    sales_year2.columns.month.values[
                        np.newaxis, :, np.newaxis],
                    sum(nonzero), axis=0) - 4,
                # YEAR
                (np.ones((sum(nonzero), LOOKBACK + 15, 1)) *
                    (sales_year2.columns.tolist()[0].year - 2015)),
                # Week day
                np.repeat(
                    sales_year2.columns.weekday.values[
                        np.newaxis, :, np.newaxis],
                    sum(nonzero), axis=0),
                # item freshness 0 if old; 1 if new
                np.repeat(
                    ((ty2 - df_first_date).astype(
                        "timedelta64[D]").astype("int") < 56
                     ).astype("int16")[nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1),
                np.repeat(
                    items["perishable"].values[
                        nonzero, np.newaxis, np.newaxis],
                    LOOKBACK + 15, axis=1)
            ], axis=2
        ).astype("int16")
        # Avoid data leaks
        x[:, -15:, 2] = 0
        x[:, -15:, 4] = 0
        if is_train:
            y = get_timespan(df_sales, ty2, 0, 16).values.astype("float64")
            x[:, -15:, 0] = y[nonzero, :15]
            return x, x_int, y[nonzero, :], sales_year2[nonzero].index
        return x, x_int, sales_year2[nonzero].index

    def fill_train_data(path_prefix, ty1, ty2, current_cnt=0):
        assert ty1.weekday() == 2
        assert ty2.weekday() == 2
        print(
            "%07d" % current_cnt,
            ty1 - timedelta(days=LOOKBACK), ty1, ty1 + timedelta(days=15),
            ty2 - timedelta(days=LOOKBACK), ty2, ty2 + timedelta(days=15))
        x_tmp, x_i_tmp, y_tmp, idx_store_item = prepare_dataset(ty1, ty2)
        if current_cnt == 0:
            x = np.memmap(
                "cache/x{}_seq.npy".format(path_prefix), mode="w+",
                order="C", dtype="float64",
                shape=(x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2]))
            x_i = np.memmap(
                "cache/x{}_i_seq.npy".format(path_prefix), mode="w+",
                order="C", dtype="int16",
                shape=(x_tmp.shape[0], x_i_tmp.shape[1], x_i_tmp.shape[2]))
            y = np.memmap(
                "cache/y{}_seq.npy".format(path_prefix), mode="w+",
                order="C", dtype="float64",
                shape=(x_tmp.shape[0], y_tmp.shape[1]))
        else:
            x = np.memmap(
                "cache/x{}_seq.npy".format(path_prefix), mode="r+",
                order="C", dtype="float64",
                shape=(current_cnt + x_tmp.shape[0], x_tmp.shape[1], x_tmp.shape[2]))
            x_i = np.memmap(
                "cache/x{}_i_seq.npy".format(path_prefix), mode="r+",
                order="C", dtype="int16",
                shape=(current_cnt + x_tmp.shape[0], x_i_tmp.shape[1], x_i_tmp.shape[2]))
            y = np.memmap(
                "cache/y{}_seq.npy".format(path_prefix), mode="r+",
                order="C", dtype="float64",
                shape=(current_cnt + x_tmp.shape[0], y_tmp.shape[1]))
        x[current_cnt:, :, :] = x_tmp
        x_i[current_cnt:, :, :] = x_i_tmp
        y[current_cnt:, :] = y_tmp
        current_cnt += x_tmp.shape[0]
        x.flush()
        x_i.flush()
        y.flush()
        return current_cnt, idx_store_item

    print("Preparing dataset...")
    # Train
    ty1 = date(2016, 6, 29) - timedelta(days=4 * 7)
    ty2 = date(2017, 6, 28) - timedelta(days=4 * 7)
    current_cnt, _ = fill_train_data("train", ty1, ty2, 0)
    for i in range(1, 6):
        delta = timedelta(days=7 * i)
        current_cnt, _ = fill_train_data(
            "train", ty1 + delta, ty2 + delta, current_cnt)

    ty1 = date(2015, 7, 1)
    ty2 = date(2016, 6, 29)
    for i in range(0, 11, 2):
        delta = timedelta(days=7 * i)
        current_cnt, _ = fill_train_data(
            "train", ty1 + delta, ty2 + delta, current_cnt)

    ty1 = date(2014, 7, 2)
    ty2 = date(2015, 7, 1)
    for i in range(0, 11, 2):
        delta = timedelta(days=7 * i)
        current_cnt, _ = fill_train_data(
            "train", ty1 + delta, ty2 + delta, current_cnt)

    print("Train count", current_cnt)

    # Validation
    ty1 = date(2016, 7, 27)
    ty2 = date(2017, 7, 26)
    current_cnt, idx_store_item = fill_train_data("val", ty1, ty2, 0)
    joblib.dump(idx_store_item, "cache/val_idx.pkl")
    print("Val count", current_cnt)

    x_test, x_i_test, idx_store_item = prepare_dataset(
        date(2016, 8, 17), date(2017, 8, 16), is_train=False)
    joblib.dump(idx_store_item, "cache/test_idx.pkl")
    x = np.memmap("cache/xtest_seq.npy", mode="w+", order="C", dtype="float64",
                  shape=(x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    x_i = np.memmap("cache/xtest_i_seq.npy", mode="w+", order="C", dtype="int16",
                    shape=(x_i_test.shape[0], x_i_test.shape[1], x_i_test.shape[2]))
    x[:, :, :] = x_test
    x_i[:, :, :] = x_i_test
    x.flush()
    x_i.flush()
    print(x_test.shape)
    return


if __name__ == "__main__":
    main()
