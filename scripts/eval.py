import argparse

import torch
from torch.utils.data import DataLoader

from detector.data import CifakeDataset
from detector.transform import get_eval_transform_resnet
from detector.inference import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="test 세트 평가")
    parser.add_argument("--data-root", default="data/raw/test")
    parser.add_argument("--ckpt", default="models/cifake_resnet18.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model, device = load_model(args.ckpt)

    dataset = CifakeDataset(root=args.data_root, transform=get_eval_transform_resnet())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"device: {device} | test 샘플: {len(dataset)}장\n")

    correct = total = 0
    # 혼동행렬: cm[정답][예측]
    cm = [[0, 0], [0, 0]]
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for t, p in zip(labels.tolist(), preds.tolist()):
            cm[t][p] += 1

    acc = correct / total
    print(f"✅ Test Accuracy: {acc:.4f} ({correct}/{total})\n")
    print("          예측:REAL  예측:FAKE")
    print(f"정답 REAL   {cm[0][0]:8d}  {cm[0][1]:8d}")
    print(f"정답 FAKE   {cm[1][0]:8d}  {cm[1][1]:8d}")


if __name__ == "__main__":
    main()
