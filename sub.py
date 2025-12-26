import torch
import pandas as pd
from model import TCNForecastModel
from data_processing import SalesDataset, add_holidays, add_oil

@torch.no_grad()
def main():
    meta = torch.load("checkpoints/meta.pt")
    model = TCNForecastModel(meta)
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()

    df = pd.read_csv("dataset/test.csv", parse_dates=["date"])
    stores = pd.read_csv("dataset/stores.csv")
    oil = pd.read_csv("dataset/oil.csv", parse_dates=["date"])
    holidays = pd.read_csv("dataset/holidays_events.csv", parse_dates=["date"])

    df["family"] = df["family"].astype("category").cat.codes
    df = df.merge(stores[["store_nbr", "cluster"]], on="store_nbr")
    
    # Add oil and holidays to test data
    df = add_holidays(df, holidays)
    df = add_oil(df, oil)

    preds = []
    history = {}

    for _, row in df.iterrows():
        key = (row.store_nbr, row.family)
        lags = history.get(key, [0.0, 0.0, 0.0])

        # Create sequence with proper shape and data types
        seq = torch.tensor(
            [[lags[0], lags[1], lags[2], row.onpromotion, row.oil, row.is_holiday]], 
            dtype=torch.float32
        )
        
        batch = {
            "seq": seq.unsqueeze(0),  # Shape: [1, 1, 6]
            "store": torch.tensor([row.store_nbr], dtype=torch.long),
            "cluster": torch.tensor([row.cluster], dtype=torch.long),
            "family": torch.tensor([row.family], dtype=torch.long),
        }

        y = model(batch).item()
        preds.append(max(0, y))
        history[key] = [y, lags[0], lags[1]]

    pd.DataFrame({"id": df["id"], "sales": preds}).to_csv("submission.csv", index=False)
    print("submission.csv created")

if __name__ == "__main__":
    main()