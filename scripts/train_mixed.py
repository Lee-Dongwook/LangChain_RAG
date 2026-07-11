"""혼합 재학습: CIFAKE(train) + OOD(art_train)을 함께 학습.

OOD 붕괴(art 0.499)를 개선하기 위해 학습 분포에 다른 생성기·도메인을 섞는다.
누수 방지를 위해 art는 scripts/split_ood.py로 나눈 art_train만 쓰고,
art_holdout / CIFAKE test 는 평가 전용으로 남긴다.

사용:
    python scripts/train_mixed.py --quick     # 로컬 빠른 플럼빙 검증(작은 subset)
    python scripts/train_mixed.py             # 전체(느림 — GPU 권장)

학습 후 평가:
    python scripts/eval.py --ckpt experiments/mixed/best.pt                          # CIFAKE test
    python scripts/eval.py --ckpt experiments/mixed/best.pt --data-root data/ood/art_holdout
"""
import argparse

from detector.training.run import run_training

CIFAKE_TRAIN = "data/raw/train"
ART_TRAIN = "data/ood/art_train"


def main():
    parser = argparse.ArgumentParser(description="CIFAKE + OOD 혼합 재학습")
    parser.add_argument("--quick", action="store_true",
                        help="작은 subset으로 빠르게 플럼빙만 검증")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out-dir", default="experiments/mixed")
    args = parser.parse_args()

    run_training(
        data_root=[CIFAKE_TRAIN, ART_TRAIN],   # ← 멀티-루트 혼합
        out_dir=args.out_dir,
        epochs=1 if args.quick else args.epochs,
        batch_size=64,
        lr=1e-4,
        freeze_backbone=False,
        max_samples=2000 if args.quick else None,
    )


if __name__ == "__main__":
    main()
