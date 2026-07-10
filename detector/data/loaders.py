import torch 
from torch.utils.data import DataLoader, Subset
from detector.data import CifakeDataset
from detector.transform import get_train_transform, get_eval_transform

def _check_class_balance(train_sub, val_sub):
    train_labels = [train_sub.dataset.samples[i][1] for i in train_sub.indices]
    val_labels = [val_sub.dataset.samples[i][1] for i in val_sub.indices]

    print("📊 [데이터 분할 밸런스 검증]")
    print(f"   - Train 세트: REAL(0) = {train_labels.count(0)}장, FAKE(1) = {train_labels.count(1)}장")
    print(f"   - Val 세트:   REAL(0) = {val_labels.count(0)}장, FAKE(1) = {val_labels.count(1)}장")


def build_loaders(root: str, train_transform=None, eval_transform=None,
                  batch_size: int = 64, val_ratio: float = 0.2, seed: int = 42):
    # transform을 밖에서 주입받는다. 안 주면 SimpleCNN용 기본값 사용(하위호환).
    if train_transform is None:
        train_transform = get_train_transform()
    if eval_transform is None:
        eval_transform = get_eval_transform()

    generator = torch.Generator().manual_seed(seed)

    full_train_dataset = CifakeDataset(root=root, transform=train_transform)
    full_val_dataset = CifakeDataset(root=root, transform=eval_transform)

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator
    )

    train_dataset = Subset(full_train_dataset, train_indices.indices)
    val_dataset = Subset(full_val_dataset, val_indices.indices)

    _check_class_balance(train_dataset, val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader



