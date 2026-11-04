"""Train ECOLOGIC's MobileNetV3-Small image classifier."""
import argparse
import copy
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return accuracy, macro_f1, y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train ECOLOGIC MobileNetV3-Small")
    parser.add_argument("--data-dir", default="data/organic_waste")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/ecologic_mobilenetv3.pth")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download pretrained weights")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{val_dir}'. Run prepare_dataset.py and inspect its output."
        )

    weights = None if args.no_pretrained else MobileNet_V3_Small_Weights.DEFAULT
    mean = [0.485, 0.456, 0.406] if weights is None else list(weights.transforms().mean)
    std = [0.229, 0.224, 0.225] if weights is None else list(weights.transforms().std)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    if train_ds.classes != val_ds.classes:
        raise ValueError(f"Class mismatch. train={train_ds.classes}; val={val_ds.classes}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must both contain images.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(train_ds.classes))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_accuracy={val_acc:.4f} | val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    val_acc, val_f1, y_val, pred_val = evaluate(model, val_loader, device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "architecture": "mobilenet_v3_small",
        "image_size": 224,
        "normalization_mean": mean,
        "normalization_std": std,
        "best_epoch": best_epoch,
        "best_val_accuracy": val_acc,
        "best_val_macro_f1": val_f1,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output)

    print(f"\nSaved best checkpoint: {args.output}")
    print(f"Best epoch: {best_epoch}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro-F1: {val_f1:.4f}")
    print("\nValidation classification report:")
    print(classification_report(y_val, pred_val, labels=list(range(len(train_ds.classes))),
                                target_names=train_ds.classes, zero_division=0))
    print("Validation confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_val, pred_val, labels=list(range(len(train_ds.classes)))))

    if os.path.isdir(test_dir) and any(os.path.isdir(os.path.join(test_dir, c)) for c in os.listdir(test_dir)):
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)
        if test_ds.classes != train_ds.classes:
            print("\nWARNING: Test classes differ from train classes; skipping test evaluation.")
        elif len(test_ds) > 0:
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_acc, test_f1, y_test, pred_test = evaluate(model, test_loader, device)
            print(f"\nHeld-out test accuracy: {test_acc:.4f}")
            print(f"Held-out test macro-F1: {test_f1:.4f}")
            print(classification_report(y_test, pred_test, labels=list(range(len(test_ds.classes))),
                                        target_names=test_ds.classes, zero_division=0))
    else:
        print("\nNo test split detected; only validation metrics are reported.")
    print("\nDo not report these metrics as general performance without documenting the dataset and split.")


if __name__ == "__main__":
    from pathlib import Path
    main()
