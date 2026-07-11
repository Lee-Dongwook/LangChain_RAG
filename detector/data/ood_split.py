"""OOD 데이터의 결정적 train/holdout 분할 (로컬·Kaggle 공용).

로컬(scripts/split_ood.py)과 Kaggle 커널이 **같은 함수**를 써야 holdout이
학습에 새어들지 않는다. seed·클래스 순회 순서를 고정해 어디서든 동일하게 나눈다.
"""
import os
import random

CLASSES = ("REAL", "FAKE")
IMG_EXTS = (".jpg", ".jpeg", ".png")


def split_ood(src, holdout_ratio: float = 0.2, seed: int = 42):
    """src/{REAL,FAKE} 를 결정적으로 분할.

    반환: {"train": {cls: [files]}, "holdout": {cls: [files]}}
    단일 rng를 CLASSES 순서대로 소비하므로 호출 환경과 무관하게 동일 분할.
    """
    rng = random.Random(seed)
    train, holdout = {}, {}
    for cls in CLASSES:
        cls_dir = os.path.join(src, cls)
        files = sorted(
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith(IMG_EXTS)
        )
        rng.shuffle(files)
        n_holdout = int(len(files) * holdout_ratio)
        holdout[cls], train[cls] = files[:n_holdout], files[n_holdout:]
    return {"train": train, "holdout": holdout}


def materialize(part, dst_root):
    """{cls: [files]} 를 dst_root/{cls}/ 에 심볼릭 링크로 실체화하고 dst_root 반환.

    원본을 복사하지 않아 디스크를 아끼며, CifakeDataset이 그대로 읽을 수 있다.
    """
    for cls, files in part.items():
        cls_dir = os.path.join(dst_root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for src in files:
            dst = os.path.join(cls_dir, os.path.basename(src))
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
    return dst_root
