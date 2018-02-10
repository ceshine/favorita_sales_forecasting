import numpy as np
from torch.utils.data import Dataset

from joblib import Memory
from models import TRAIN_PERIODS

memory = Memory(cachedir="cache/", verbose=1)

MEMMAP_SHAPES = {
    "train": [(2459797, 71, 6), (2459797, 71, 16), (2459797, 16)],
    "val": [(154764, 71, 6), (154764, 71, 16), (154764, 16)],
    "test": [(154571, 71, 6), (154571, 71, 16)]
}


def read_dataset():
    x_train = "cache/xtrain_seq.npy"
    x_i_train = "cache/xtrain_i_seq.npy"
    y_train = "cache/ytrain_seq.npy"
    x_val = "cache/xval_seq.npy"
    x_i_val = "cache/xval_i_seq.npy"
    y_val = "cache/yval_seq.npy"
    x_test = "cache/xtest_seq.npy"
    x_i_test = "cache/xtest_i_seq.npy"

    train_dataset = FavoritaDataset(
        MEMMAP_SHAPES["train"], x_train, x_i_train, y_train, train_periods=TRAIN_PERIODS)
    val_dataset = FavoritaDataset(
        MEMMAP_SHAPES["val"], x_val, x_i_val, y_val, train_periods=TRAIN_PERIODS,
        reference_dataset=train_dataset)
    test_dataset = FavoritaDataset(
        MEMMAP_SHAPES["test"], x_test, x_i_test, train_periods=TRAIN_PERIODS,
        reference_dataset=train_dataset)
    return train_dataset, val_dataset, test_dataset


@memory.cache
def fit_residual_series_stats(x, shape, train_periods, cols=[0, 1, 2, 3, 4, 5], get_stds=False):
    series = np.memmap(
        x, mode="r", order="C", dtype="float64", shape=shape)
    means = np.zeros((series.shape[0], len(cols)))
    stds = np.zeros(len(cols))
    for i, col in enumerate(cols):
        data = series[:, :, col]
        means[:, i] = np.mean(data[:, :train_periods], axis=1)
        if get_stds:
            data = data - np.expand_dims(means[:, i], 1)
            stds[i] = np.std(
                data[:, :train_periods].reshape(-1))
    mean_of_means = np.mean(means, axis=0)
    std_of_means = np.std(means, axis=0)
    return means, stds, mean_of_means, std_of_means


class FavoritaDataset(Dataset):
    def __init__(self, shapes, x, x_i, y=None, reference_dataset=None, train_periods=TRAIN_PERIODS, clip_low=-3, clip_high=3):
        """
        args
        ----
            x:
            y:   pass False for test dataset
            split_point: where to split data for encoder/decoder
        """
        self.reference_dataset = reference_dataset
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.series = np.memmap(
            x, mode="r", order="C", dtype="float64", shape=shapes[0])
        self.series_i = np.memmap(
            x_i, mode="r", order="C", dtype="int16", shape=shapes[1])
        self.train_periods = train_periods
        if y is None:
            self.is_train = False
        else:
            self.is_train = True
            self.y = np.memmap(y, mode="r", order="C",
                               dtype="float64", shape=shapes[2])
        print("Fitting residual stats for {}...".format(x))
        self.means, self.stds, self.mean_of_means, self.std_of_means = fit_residual_series_stats(
            x, shapes[0], train_periods, get_stds=(reference_dataset is None)
        )
        if reference_dataset is not None:
            self.stds = reference_dataset.stds

    def __len__(self):
        return len(self.series)

    def normalize_series(self, idx):
        raw_data = self.series[idx, :, :6]
        residual_data = (
            raw_data - self.means[idx, np.newaxis, :]) / self.stds[np.newaxis, :]
        return np.clip(np.nan_to_num(
            residual_data), self.clip_low, self.clip_high), raw_data

    def trim_series(self, s1, s2):
        """Not doing anything useful here. Just for backward compatibility."""
        assert len(s1) == len(s2)
        return s1, s2

    def derive_features(self, idx):
        feat = np.zeros(6)
        cnt = 0
        series = np.trim_zeros(self.series[idx, :self.train_periods, 0])
        # Yearly correlation
        y2, y1 = self.trim_series(
            # year 2
            self.series[idx, 1:self.train_periods, 0],
            # year 1
            self.series[idx, :(self.train_periods - 1), 1]
        )
        if len(y2) > 5 and np.std(y1) > 0.01 and np.std(y2) > 0.01:
            feat[cnt] = np.corrcoef(y2, y1)[0, 1]
        cnt += 1
        # Cluster sales Yearly correlation
        y2, y1 = self.trim_series(
            # year 2 item sales
            self.series[idx, 1:self.train_periods, 0],
            # year 1 cluster item sales
            self.series[idx, :(self.train_periods - 1), 3]
        )
        if len(y2) > 5 and np.std(y1) > 0.01 and np.std(y2) > 0.01:
            feat[cnt] = np.corrcoef(y2, y1)[0, 1]
        cnt += 1
        # Item class yearly correlation
        # y2, y1 = self.trim_series(
        #     # year 2 item sales
        #     numeric_series[1:self.train_periods, 0],
        #     # year 1 item class sales
        #     numeric_series[:(self.train_periods - 1), 5]
        # )
        # if len(y2) > 5 and np.std(y1) > 0.01 and np.std(y2) > 0.01:
        #     feat[cnt] = np.corrcoef(y2, y1)[0, 1]
        # cnt += 1
        # Year 2 cluster sales mean
        feat[cnt] = (self.means[idx, 2] - self.mean_of_means[2]) / \
            self.std_of_means[2]
        cnt += 1
        # Year 1 cluster sales mean
        feat[cnt] = (self.means[idx, 3] - self.mean_of_means[3]) / \
            self.std_of_means[3]
        cnt += 1
        # # Year 2 item class mean
        # feat[cnt] = (self.means[idx, 4] - self.mean_of_means[4]) / \
        #     self.std_of_means[4]
        # cnt += 1
        # # Year 1 item class mean
        # feat[cnt] = (self.means[idx, 5] - self.mean_of_means[5]) / \
        #     self.std_of_means[5]
        # cnt += 1
        # year 2 mean
        feat[cnt] = (self.means[idx, 0] - self.mean_of_means[0]) / \
            self.std_of_means[0]
        cnt += 1
        # year 1 mean
        feat[cnt] = (self.means[idx, 1] - self.mean_of_means[1]) / \
            self.std_of_means[1]
        cnt += 1
        assert cnt == feat.shape[0]
        return np.nan_to_num(feat)

    def __getitem__(self, idx):
        numeric_series, raw_series = self.normalize_series(idx)
        integer_series = self.series_i[idx, :, :].__array__().astype("int32")
        if self.is_train:
            return (
                numeric_series.astype("float32"),
                self.derive_features(idx).astype("float32"),
                np.concatenate([
                    integer_series[:, :-1],
                    # year 2 zero sale
                    (raw_series[:, :1] == 0).astype("int32"),
                    integer_series[:, -1:]
                ], axis=1),
                self.means[idx].astype("float32"),
                self.y[idx, :].__array__().astype("float32")
            )
        else:
            return (
                numeric_series.astype("float32"),
                self.derive_features(idx).astype("float32"),
                np.concatenate([
                    integer_series[:, :-1],
                    # year 2 zero sale
                    (raw_series[:, :1] == 0).astype("int32"),
                    integer_series[:, -1:],
                ], axis=1),
                self.means[idx].astype("float32")
            )
