
"""
train_resnet.py

PharmaChain - Member 4: Counterfeit Detector

ResNet-18 transfer learning:
Genuine vs Counterfeit Packaging

NOTE:
Model predictions are experimental and must not be used
to establish medicine authenticity or patient safety.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

IMG = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NUM_CLASSES = 2


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed=0):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------

def make_transforms():
    """Create training and evaluation transforms."""

    train_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),

        transforms.RandomRotation(3),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.03, 0.03)
        ),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD)
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG, IMG)),

        transforms.ToTensor(),

        transforms.Normalize(MEAN, STD)
    ])

    return train_tf, eval_tf


# --------------------------------------------------
# MODEL CREATION
# --------------------------------------------------

def build_model(pretrained=True, unfreeze="none",
                num_classes=NUM_CLASSES):
    """Build a ResNet-18 transfer learning model."""

    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.resnet18(weights=weights)

    # Freeze all backbone parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Optionally unfreeze selected layers
    if unfreeze == "layer4":

        for param in model.layer4.parameters():
            param.requires_grad = True

    elif unfreeze == "all":

        for param in model.parameters():
            param.requires_grad = True

    # Replace final classification layer
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        num_classes
    )

    return model


# --------------------------------------------------
# TRAINING / VALIDATION
# --------------------------------------------------

def run_epoch(model, loader, loss_fn, device,
              optimizer=None):
    """Run one training or validation epoch."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.set_grad_enabled(is_training):

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            if is_training:

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            batch_size = labels.size(0)

            loss_sum += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    if total == 0:
        raise ValueError(
            "The DataLoader is empty. Check your dataset."
        )

    average_loss = loss_sum / total
    accuracy = correct / total

    return average_loss, accuracy


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

def evaluate_test(model, loader, class_names, device):
    """Evaluate the trained model on test data."""

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())

            y_pred.extend(
                predictions.cpu().tolist()
            )

    if not y_true:
        raise ValueError("Test dataset is empty.")

    print("\nTest Set Results")
    print("------------------------------")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            digits=3,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    print(
        "Confusion Matrix "
        "(rows = true, columns = predicted):"
    )

    print(cm)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }


# --------------------------------------------------
# DATASET VALIDATION
# --------------------------------------------------

def validate_dataset(data_dir):

    required_dirs = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val")
    ]

    for directory in required_dirs:

        if not os.path.isdir(directory):
            raise FileNotFoundError(
                "Dataset directory not found: "
                + directory
            )

    train_classes = sorted(
        name for name in os.listdir(required_dirs[0])
        if os.path.isdir(
            os.path.join(required_dirs[0], name)
        )
    )

    val_classes = sorted(
        name for name in os.listdir(required_dirs[1])
        if os.path.isdir(
            os.path.join(required_dirs[1], name)
        )
    )

    if len(train_classes) != NUM_CLASSES:
        raise ValueError(
            "Expected exactly 2 training classes. Found: "
            + str(train_classes)
        )

    if train_classes != val_classes:
        raise ValueError(
            "Training and validation class folders must match."
        )

    return train_classes


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train ResNet-18 for packaging image classification."
    )

    parser.add_argument(
        "--data",
        default="data",
        help="Dataset directory"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--unfreeze",
        choices=["none", "layer4", "all"],
        default="none",
        help="Which backbone layers to train"
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not download ImageNet pretrained weights"
    )

    parser.add_argument(
        "--out",
        default="model.pt",
        help="Output model filename"
    )

    parser.add_argument(
        "--metrics",
        default="metrics.json",
        help="Output metrics filename"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero")

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    if args.lr <= 0:
        parser.error("--lr must be greater than zero")

    # Set seeds
    set_seed(args.seed)

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n===================================")
    print("PHARMACHAIN - RESNET18 TRAINING")
    print("===================================")

    print("Device:", device)

    # Validate dataset
    class_names = validate_dataset(args.data)

    print("Classes:", class_names)

    # Create transforms
    train_tf, eval_tf = make_transforms()

    # Load datasets
    train_ds = datasets.ImageFolder(
        os.path.join(args.data, "train"),
        transform=train_tf
    )

    val_ds = datasets.ImageFolder(
        os.path.join(args.data, "val"),
        transform=eval_tf
    )

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty.")

    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty.")

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            "Training and validation class mappings differ."
        )

    print(
        "\nTraining images: %d"
        % len(train_ds)
    )

    print(
        "Validation images: %d"
        % len(val_ds)
    )

    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Build model
    model = build_model(
        pretrained=not args.no_pretrained,
        unfreeze=args.unfreeze,
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    # Trainable parameters
    params = [
        param
        for param in model.parameters()
        if param.requires_grad
    ]

    if not params:
        raise ValueError(
            "No trainable parameters found."
        )

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        params,
        lr=args.lr
    )

    loss_fn = nn.CrossEntropyLoss()

    # Best model tracking
    best_acc = -1.0
    best_state = None

    # Training loop
    print("\nStarting training...\n")

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model,
            train_dl,
            loss_fn,
            device,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_dl,
            loss_fn,
            device
        )

        elapsed = time.time() - start_time

        print(
            "Epoch %d/%d | "
            "Train Loss: %.3f | "
            "Train Acc: %.3f | "
            "Val Loss: %.3f | "
            "Val Acc: %.3f | "
            "Time: %.1fs"
            % (
                epoch,
                args.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                elapsed
            )
        )

        # Save best validation model
        if val_acc > best_acc:

            best_acc = val_acc

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model.")

    # Restore best model
    model.load_state_dict(best_state)

    # Save model checkpoint
    output_dir = os.path.dirname(
        os.path.abspath(args.out)
    )

    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "state_dict": best_state,
            "class_names": class_names,
            "img_size": IMG,
            "best_val_acc": best_acc
        },
        args.out
    )

    print("\nBest validation accuracy: %.3f" % best_acc)

    print("Saved model to:", args.out)

    # Test evaluation
    test_dir = os.path.join(
        args.data,
        "test"
    )

    if os.path.isdir(test_dir):

        test_ds = datasets.ImageFolder(
            test_dir,
            transform=eval_tf
        )

        if test_ds.classes != class_names:
            raise ValueError(
                "Test classes do not match training classes."
            )

        if len(test_ds) == 0:
            raise ValueError("Test dataset is empty.")

        test_dl = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda")
        )

        metrics = evaluate_test(
            model,
            test_dl,
            test_ds.classes,
            device
        )

        metrics["best_val_acc"] = best_acc

        metrics_dir = os.path.dirname(
            os.path.abspath(args.metrics)
        )

        os.makedirs(metrics_dir, exist_ok=True)

        with open(
            args.metrics,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metrics,
                f,
                indent=2
            )

        print("\nSaved metrics to:", args.metrics)

    else:

        print(
            "\nTest directory not found. "
            "Skipping test evaluation."
        )

    print("\n===================================")
    print("TRAINING COMPLETED")
    print("===================================")


# --------------------------------------------------
# PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
