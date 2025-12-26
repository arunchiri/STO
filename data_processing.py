import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# ------------------------
# RMSLE-aligned loss
# ------------------------
def rmsle_loss(pred, y):
    return torch.sqrt(torch.mean((pred - y) ** 2))


# ------------------------
# External features
# ------------------------
def add_oil(df, oil):
    oil = oil.rename(columns={"dcoilwtico": "oil"})
    oil["oil"] = oil["oil"].ffill()
    return df.merge(oil, on="date", how="left").ffill()


def add_holidays(df, holidays):
    h = holidays[holidays["transferred"] == False]
    h = h.groupby("date").size().reset_index(name="is_holiday")
    return df.merge(h, on="date", how="left").fillna(0)


# ------------------------
# Dataset
# ------------------------
class SalesDataset(Dataset):
    def __init__(self, df, seq_cols, train=True):
        self.seq = torch.tensor(
            df[seq_cols].values,
            dtype=torch.float32
        ).view(len(df), 1, len(seq_cols))

        self.store = torch.tensor(df["store_nbr"].values, dtype=torch.long)
        self.cluster = torch.tensor(df["cluster"].values, dtype=torch.long)
        self.family = torch.tensor(df["family"].values, dtype=torch.long)

        self.y = None
        if train:
            self.y = torch.tensor(df["sales"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.store)

    def __getitem__(self, i):
        out = {
            "seq": self.seq[i],
            "store": self.store[i],
            "cluster": self.cluster[i],
            "family": self.family[i],
        }
        if self.y is not None:
            out["y"] = self.y[i]
        return out


# ------------------------
# Main loader
# ------------------------
def load_datasets(data_dir):
    df = pd.read_csv(f"{data_dir}/train.csv", parse_dates=["date"])
    stores = pd.read_csv(f"{data_dir}/stores.csv")
    oil = pd.read_csv(f"{data_dir}/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(f"{data_dir}/holidays_events.csv", parse_dates=["date"])

    # Encode family
    df["family"] = df["family"].astype("category").cat.codes

    # Merge store clusters
    df = df.merge(stores[["store_nbr", "cluster"]], on="store_nbr")

    # External features
    df = add_oil(df, oil)
    df = add_holidays(df, holidays)

    # Lag features
    for l in [7, 14, 28]:
        df[f"lag_{l}"] = df.groupby(
            ["store_nbr", "family"]
        )["sales"].shift(l)

    df = df.dropna().reset_index(drop=True)

    # LOG TARGET (CRITICAL)
    df["sales"] = np.log1p(df["sales"])

    seq_cols = ["lag_7", "lag_14", "lag_28", "onpromotion", "oil", "is_holiday"]

    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

    meta = {
        "seq_cols": seq_cols,
        "n_stores": df["store_nbr"].nunique() + 1,
        "n_clusters": df["cluster"].nunique() + 1,
        "n_families": df["family"].nunique() + 1,
        "seq_features": len(seq_cols),
    }

    return (
        SalesDataset(train_df, seq_cols, True),
        SalesDataset(val_df, seq_cols, True),
        meta,
        df  # return full df for warm-start
    )
