"""OOD(art) 데이터를 train / holdout 으로 결정적 분할 (로컬용 CLI).

혼합 재학습의 데이터 누수를 막기 위해, art를 학습용(train)과 절대 학습에
쓰지 않는 평가용(holdout)으로 나눈다. 실제 분할 로직은 로컬·Kaggle 공용
detector.data.ood_split 에 있어, 어디서 나눠도 결과가 동일하다.

사용:
    python scripts/split_ood.py
    python scripts/split_ood.py --src data/ood/art/Data --out data/ood --holdout-ratio 0.2
"""
import argparse
import os

from detector.data.ood_split import split_ood, materialize


def main():
    parser = argparse.ArgumentParser(description="OOD 데이터 train/holdout 분할")
    parser.add_argument("--src", default="data/ood/art/Data")
    parser.add_argument("--out", default="data/ood")
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="art", help="출력 폴더 접두사")
    args = parser.parse_args()

    split = split_ood(args.src, holdout_ratio=args.holdout_ratio, seed=args.seed)

    train_root = materialize(split["train"], os.path.join(args.out, f"{args.prefix}_train"))
    holdout_root = materialize(split["holdout"], os.path.join(args.out, f"{args.prefix}_holdout"))

    for cls in split["train"]:
        print(f"{cls}: train {len(split['train'][cls])} / holdout {len(split['holdout'][cls])}")
    print(f"\n✅ train  → {train_root}")
    print(f"✅ holdout → {holdout_root}  (학습 금지, 평가 전용)")


if __name__ == "__main__":
    main()
