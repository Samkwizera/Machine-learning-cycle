import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

try:
    from .preprocessing import NUM_CLASSES, CLASS_NAMES
except ImportError:
    from preprocessing import NUM_CLASSES, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v2(weights=weights)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return net.to(DEVICE)


def train_model(net, train_loader, val_loader, epochs=15, lr=1e-3,
                weight_decay=1e-4, patience=3, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    stale = 0

    for epoch in range(epochs):
        t0 = time.time()
        net.train()
        run_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_loss, train_acc = run_loss / total, correct / total

        val_loss, val_acc = evaluate_loss(net, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            print(f"epoch {epoch + 1:02d}/{epochs} "
                  f"| train_loss {train_loss:.4f} acc {train_acc:.3f} "
                  f"| val_loss {val_loss:.4f} acc {val_acc:.3f} "
                  f"| {time.time() - t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(net.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch + 1}")
                break

    net.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_loss(net, loader, criterion):
    net.eval()
    run_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = net(images)
        run_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return run_loss / total, correct / total


@torch.no_grad()
def predict_loader(net, loader):
    net.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        outputs = net(images.to(DEVICE))
        y_pred.extend(outputs.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def save_model(net, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "class_names": CLASS_NAMES}, path)


def load_model(path, num_classes=NUM_CLASSES):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    net = build_model(num_classes=num_classes, pretrained=False)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


def retrain_model(checkpoint_path, train_loader, val_loader, epochs=8, lr=5e-4, **kwargs):
    # start from the saved weights and fine-tune, with a smaller lr than the
    # first fit so we don't wipe what the model already learned
    net = load_model(checkpoint_path)
    history = train_model(net, train_loader, val_loader, epochs=epochs, lr=lr, **kwargs)
    return net, history
