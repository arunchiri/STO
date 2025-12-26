import argparse
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model import TCNForecastModel
from data_processing import load_datasets, rmsle_loss


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        preds = model(batch)
        loss = rmsle_loss(preds, batch["y"])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch["y"])

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)

        preds = model(batch)
        loss = rmsle_loss(preds, batch["y"])
        total_loss += loss.item() * len(batch["y"])

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    # ==============================
    # Training start
    # ==============================
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print(f"Training started at: {start_dt}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 60 + "\n")

    # Load data
    train_ds, val_ds, meta, _ = load_datasets(args.data_dir)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print("=" * 60)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size
    )

    # Model
    model = TCNForecastModel(meta).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)

    best_val = float("inf")
    epochs_no_improve = 0

    # ==============================
    # Training loop
    # ==============================
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, args.device, epoch
        )
        val_loss = eval_epoch(
            model, val_loader, args.device
        )

        improved = val_loss < (best_val - args.min_delta)

        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best.pt")
            torch.save(meta, "checkpoints/meta.pt")
            flag = " *"
        else:
            epochs_no_improve += 1
            flag = ""

        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSLE: {train_loss:.4f} | "
            f"Val RMSLE: {val_loss:.4f} | "
            f"NoImprove: {epochs_no_improve}/{args.patience}"
            f"{flag}"
        )

        # ---- Early stopping ----
        if epochs_no_improve >= args.patience:
            print(
                f"\nEarly stopping triggered at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)."
            )
            break

    # ==============================
    # Training end
    # ==============================
    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elapsed = int(end_time - start_time)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    print("\n" + "=" * 60)
    print(f"Training ended at: {end_dt}")
    if hours > 0:
        print(f"Total training time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Total training time: {minutes}m {seconds}s")
    else:
        print(f"Total training time: {seconds}s")
    print(f"Best validation RMSLE: {best_val:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
