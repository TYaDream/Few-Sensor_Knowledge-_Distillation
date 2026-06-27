import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from data import ZhangGasDataset
from model import InformerClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(n_samples, seed, train_ratio=0.6, val_ratio=0.2):
    indices = list(range(n_samples))
    random.Random(seed).shuffle(indices)
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def build_loaders(dataset, split, batch_size):
    return {
        name: DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=(name == "train"))
        for name, idx in split.items()
    }


def make_model(input_dim, hidden_dim, num_layers, target_dim, rnn_type, dropout):
    return InformerClassifier(
        input_dim=input_dim,
        d_model=hidden_dim,
        num_layers=num_layers,
        output_dim=hidden_dim,
        target_dim=target_dim,
        dropout=dropout,
        rnn_type=rnn_type,
    ).to(DEVICE)


def evaluate(model, loader, label_scale):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(torch.float32).to(DEVICE)
            y = y.to(torch.float32).to(DEVICE)
            pred = model(x)
            preds.append((pred * label_scale).cpu().numpy())
            labels.append((y * label_scale).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return {
        "rmse": float(np.sqrt(mean_squared_error(labels, preds))),
        "r2": float(r2_score(labels, preds)),
    }


def train_teacher(model, loaders, epochs, lr, label_scale, print_every):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loaders["train"]:
            x = x.to(torch.float32).to(DEVICE)
            y = y.to(torch.float32).to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_rmse = evaluate(model, loaders["val"], label_scale)["rmse"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if print_every and epoch % print_every == 0:
            train_loss = total_loss / len(loaders["train"].dataset)
            print(f"teacher epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def train_student(student, teacher, loaders, epochs, lr, alpha, label_scale, print_every):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = nn.MSELoss()
    teacher.eval()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        student.train()
        total_loss = 0.0
        for x, y in loaders["train"]:
            x = x.to(torch.float32).to(DEVICE)
            y = y.to(torch.float32).to(DEVICE)
            with torch.no_grad():
                teacher_pred = teacher(x)
            pred = student(x)
            supervised_loss = criterion(pred, y)
            distill_loss = criterion(pred, teacher_pred)
            loss = (1.0 - alpha) * supervised_loss + alpha * distill_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        val_rmse = evaluate(student, loaders["val"], label_scale)["rmse"]
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if print_every and epoch % print_every == 0:
            train_loss = total_loss / len(loaders["train"].dataset)
            print(f"student epoch={epoch} train_loss={train_loss:.6f} val_rmse={val_rmse:.6f}")

    if best_state is not None:
        student.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser(description="Minimal teacher-student training for Zhang gas sensor regression.")
    parser.add_argument("--data-root", required=True, help="Dataset root. Subdirectories should be named by comma-separated labels.")
    parser.add_argument("--output-dir", default="results/basic")
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--teacher-dim", type=int, default=64)
    parser.add_argument("--student-dim", type=int, default=32)
    parser.add_argument("--teacher-layers", type=int, default=2)
    parser.add_argument("--student-layers", type=int, default=1)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for teacher prediction loss.")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--print-every", type=int, default=10)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ZhangGasDataset(args.data_root, sequence_length=args.sequence_length, fit_stats=False)
    split = split_indices(len(dataset), args.seed)
    dataset.fit_sensor_stats(split["train"])
    loaders = build_loaders(dataset, split, args.batch_size)
    label_scale = dataset.label_scale.to(DEVICE)

    input_dim = 4
    target_dim = 3
    teacher = make_model(input_dim, args.teacher_dim, args.teacher_layers, target_dim, args.rnn_type, args.dropout)
    student = make_model(input_dim, args.student_dim, args.student_layers, target_dim, args.rnn_type, args.dropout)

    train_teacher(teacher, loaders, args.teacher_epochs, args.lr, label_scale, args.print_every)
    train_student(student, teacher, loaders, args.student_epochs, args.lr, args.distill_alpha, label_scale, args.print_every)

    metrics = {
        "teacher": evaluate(teacher, loaders["test"], label_scale),
        "student": evaluate(student, loaders["test"], label_scale),
        "config": vars(args),
    }
    torch.save(teacher.state_dict(), output_dir / "teacher.pt")
    torch.save(student.state_dict(), output_dir / "student.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
