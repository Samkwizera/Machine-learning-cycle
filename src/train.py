import argparse
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from . import data_acquisition, model as model_mod
    from .preprocessing import get_dataloader
except ImportError:
    import data_acquisition
    import model as model_mod
    from preprocessing import get_dataloader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_PATH = ROOT / "models" / "flowers_model.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_acquisition.acquire()

    train_loader = get_dataloader(DATA / "train", batch_size=args.batch, train=True)
    val_loader = get_dataloader(DATA / "val", batch_size=args.batch, train=False)
    test_loader = get_dataloader(DATA / "test", batch_size=args.batch, train=False)

    net = model_mod.build_model(pretrained=True)
    model_mod.train_model(net, train_loader, val_loader,
                          epochs=args.epochs, lr=args.lr, weight_decay=1e-4, patience=3)

    y_true, y_pred = model_mod.predict_loader(net, test_loader)
    print("\ntest metrics:")
    print(f"  accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  recall   : {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"  f1       : {f1_score(y_true, y_pred, average='macro'):.4f}")

    model_mod.save_model(net, MODEL_PATH)
    print(f"\nsaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
