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


def build_loaders(root, train_transform=None, eval_transform=None,
                  batch_size: int = 64, val_ratio: float = 0.2, seed: int = 42,
                  max_samples: int = None):
    # transform을 밖에서 주입받는다. 안 주면 SimpleCNN용 기본값 사용(하위호환).
    # root는 단일 경로(str) 또는 여러 경로(list) — 여러 개면 혼합 학습.
    # max_samples: 지정 시 전체에서 그만큼만 뽑아 학습(로컬 빠른 검증용).
    if train_transform is None:
        train_transform = get_train_transform()
    if eval_transform is None:
        eval_transform = get_eval_transform()

    generator = torch.Generator().manual_seed(seed)

    full_train_dataset = CifakeDataset(root=root, transform=train_transform)
    full_val_dataset = CifakeDataset(root=root, transform=eval_transform)

    total_size = len(full_train_dataset)

    # 전체 인덱스를 섞어 pool을 만들고(필요 시 max_samples로 자름) train/val로 분할.
    # random_split 대신 인덱스를 직접 다뤄 max_samples·혼합 root에서도 밸런스 검증이 동작.
    pool = torch.randperm(total_size, generator=generator).tolist()
    if max_samples is not None:
        pool = pool[:max_samples]

    val_size = int(len(pool) * val_ratio)
    val_idx, train_idx = pool[:val_size], pool[val_size:]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)

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



