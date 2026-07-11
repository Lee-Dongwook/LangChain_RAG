"""OOD(art) 데이터를 train / holdout 으로 결정적 분할.

혼합 재학습의 데이터 누수를 막기 위해, art를 학습용(train)과
절대 학습에 쓰지 않는 평가용(holdout)으로 나눈다. 원본을 복사하지 않고
심볼릭 링크만 만들어 디스크를 아끼며, seed 고정으로 매번 동일하게 분할된다.

사용:
    python scripts/split_ood.py
    python scripts/split_ood.py --src data/ood/art/Data --out data/ood --holdout-ratio 0.2
"""
import argparse
import os
import random

CLASSES = ("REAL", "FAKE")
IMG_EXTS = (".jpg", ".jpeg", ".png")


def _link(files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for src in files:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)


def main():
    parser = argparse.ArgumentParser(description="OOD 데이터 train/holdout 분할")
    parser.add_argument("--src", default="data/ood/art/Data")
    parser.add_argument("--out", default="data/ood")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="art", help="출력 폴더 접두사")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train_root = os.path.join(args.out, f"{args.prefix}_train")
    holdout_root = os.path.join(args.out, f"{args.prefix}_holdout")

    for cls in CLASSES:
        src_dir = os.path.join(args.src, cls)
        files = sorted(
            os.path.join(src_dir, f) for f in os.listdir(src_dir)
            if f.lower().endswith(IMG_EXTS)
        )
        rng.shuffle(files)
        n_holdout = int(len(files) * args.holdout_ratio)
        holdout, train = files[:n_holdout], files[n_holdout:]

        _link(train, os.path.join(train_root, cls))
        _link(holdout, os.path.join(holdout_root, cls))
        print(f"{cls}: train {len(train)} / holdout {len(holdout)}")

    print(f"\n✅ train  → {train_root}")
    print(f"✅ holdout → {holdout_root}  (학습 금지, 평가 전용)")


if __name__ == "__main__":
    main()
