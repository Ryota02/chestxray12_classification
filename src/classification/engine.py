import json

from pathlib import Path

import torch
import torch.nn as nn

from src.classification.metrics import compute_multiclass_metrics


def build_optimizer(model, cfg):
    train_cfg = cfg["train"]

    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


@torch.no_grad()
def evaluate_classifier(
    model,
    loader,
    device,
    class_names,
):
    model.eval()

    y_true = []
    y_pred = []
    y_probability = []
    sample_indices = []

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        y_probability.extend(
            probabilities.cpu().numpy().tolist()
        )
        sample_indices.extend(
            indices.cpu().numpy().tolist()
        )

    metrics = compute_multiclass_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_probability=y_probability,
        class_names=class_names,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probability": y_probability,
        "sample_indices": sample_indices,
    }


def fit_classifier(
    model,
    loaders,
    cfg,
    device,
    class_names,
    output_dir,
    metadata,
):
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    train_cfg = cfg["train"]

    epochs = int(train_cfg.get("epochs", 30))
    monitor = train_cfg.get("monitor", "f1_macro")

    best_score = -float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += int(
                (predictions == labels).sum().item()
            )
            total_samples += batch_size

        train_loss = total_loss / total_samples
        train_accuracy = total_correct / total_samples

        # Validation
        val_result = evaluate_classifier(
            model=model,
            loader=loaders["val"],
            device=device,
            class_names=class_names,
        )

        val_metrics = val_result["metrics"]
        monitor_score = val_metrics[monitor]

        if monitor_score is None:
            raise RuntimeError(
                f"Monitor metric '{monitor}' is None."
            )

        scheduler.step(monitor_score)

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['f1_macro']:.4f} | "
            f"Val Macro-AUC: {val_metrics['roc_auc_macro']} | "
            f"LR: {current_lr:.2e}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
                "val_metrics": val_metrics,
                "lr": current_lr,
            }
        )

        # Best model
        if monitor_score > best_score:
            best_score = float(monitor_score)

            checkpoint = {
                "epoch": epoch,
                "best_score": best_score,
                "monitor": monitor,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "class_names": class_names,
                **metadata,
            }

            torch.save(
                checkpoint,
                checkpoint_dir / "best_model.pth",
            )

            print("[SAVE] Best model")

    history_path = output_dir / "history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    return history