import os
import torch
import torch.nn as nn

from detector.data import build_loaders
from detector.models.simple_cnn import SimpleCNN
from detector.training.loop import train_one_epoch, evaluate


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")       
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    train_loader, val_loader = build_loaders(root="data/raw/train", batch_size=64)

    model = SimpleCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("experiments", exist_ok=True)
    best_val_acc = 0.0
    epochs = 5

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        print(f"[{epoch}/{epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "experiments/best.pt")
            print(f"  ✅ best 갱신 → 저장 (val_acc={va_acc:.4f})")

if __name__ == "__main__":
    main()
