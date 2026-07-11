import os
import json

import torch
import torch.nn as nn

from detector.data import build_loaders
from detector.models.resnet_finetune import build_resnet18
from detector.transform import get_train_transform_resnet, get_eval_transform_resnet
from detector.training.loop import train_one_epoch, evaluate


def pick_device():
    """cuda(Kaggle) > mps(맥) > cpu 순으로 선택."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_training(data_root, out_dir="experiments", epochs=3, batch_size=64,
                 lr=1e-4, freeze_backbone=False, seed=42, max_samples=None):
    """
    로컬(M2)과 Kaggle(GPU)이 공유하는 단일 학습 진입 함수.
    경로(data_root, out_dir)만 바꿔 부르면 어디서든 동일하게 동작한다.
    결과: out_dir/best.pt (best val acc 기준), out_dir/metrics.json (전체 로그).
    """
    device = pick_device()
    print(f"device: {device}")
    torch.manual_seed(seed)

    train_loader, val_loader = build_loaders(
        root=data_root,
        train_transform=get_train_transform_resnet(),
        eval_transform=get_eval_transform_resnet(),
        batch_size=batch_size,
        max_samples=max_samples,
    )

    model = build_resnet18(num_classes=2, pretrained=True,
                           freeze_backbone=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch}/{epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"  ✅ best 갱신 → 저장 (val_acc={va_acc:.4f})")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)

    print(f"🏁 done. best_val_acc={best_val_acc:.4f}")
    return best_val_acc
