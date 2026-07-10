from detector.training.run import run_training


def main():
    # 로컬(M2) 실행: 로컬 데이터 경로 사용
    run_training(
        data_root="data/raw/train",
        out_dir="experiments",
        epochs=3,
        batch_size=64,
        lr=1e-4,
        freeze_backbone=False,
    )


if __name__ == "__main__":
    main()
