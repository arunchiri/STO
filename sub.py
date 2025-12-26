import torch
import pandas as pd
import numpy as np
from model import TCNForecastModel
from data_processing import add_oil, add_holidays

@torch.no_grad()
def main():
    meta = torch.load("checkpoints/meta.pt")
    model = TCNForecastModel(meta)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()

    train = pd.read_csv("dataset/train.csv", parse_dates=["date"])
    test = pd.read_csv("dataset/test.csv", parse_dates=["date"])
    stores = pd.read_csv("dataset/stores.csv")
    oil = pd.read_csv("dataset/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("dataset/holidays_events.csv", parse_dates=["date"])

    train["family"] = train["family"].astype("category").cat.codes
    test["family"] = test["family"].astype("category").cat.codes

    train = train.merge(stores[["store_nbr", "cluster"]], on="store_nbr")
    test = test.merge(stores[["store_nbr", "cluster"]], on="store_nbr")

    train = add_oil(train, oil)
    train = add_holidays(train, holidays)
    test = add_oil(test, oil)
    test = add_holidays(test, holidays)

    history = {}
    for (s, f), g in train.groupby(["store_nbr", "family"]):
        g = g.sort_values("date")
        history[(s, f)] = list(g["sales"].iloc[-28:])

    preds = []

    for _, r in test.iterrows():
        key = (r.store_nbr, r.family)
        h = history[key]

        seq = torch.tensor([[
            h[-7], h[-14], h[-28],
            r.onpromotion,
            r.oil,
            r.is_holiday
        ]], dtype=torch.float32).view(1, 1, -1)

        batch = {
            "seq": seq,
            "store": torch.tensor([r.store_nbr]),
            "cluster": torch.tensor([r.cluster]),
            "family": torch.tensor([r.family]),
        }

        y_log = model(batch).item()
        y = np.expm1(y_log)

        preds.append(max(0, y))
        h.append(y)

    pd.DataFrame({"id": test["id"], "sales": preds}).to_csv("submission.csv", index=False)
    print("submission.csv created")

if __name__ == "__main__":
    main()
