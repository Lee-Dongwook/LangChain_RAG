"""
Kaggle 커널에서 실행되는 스크립트 (GitHub Actions가 `kaggle kernels push`로 올림).

동작:
  1) GitHub에서 detector 패키지를 pip 설치 (리포 public이라 토큰 불필요)
  2) /kaggle/input 에 마운트된 CIFAKE 데이터에서 train 폴더 자동 탐색
  3) run_training() 호출 → /kaggle/working 에 best.pt, metrics.json 저장
     (kaggle kernels output 으로 회수 가능)
"""
import os
import subprocess
import sys

REPO = "git+https://github.com/Lee-Dongwook/LangChain_RAG.git@main"


def install_detector():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", REPO],
        check=True,
    )


def find_train_dir(base="/kaggle/input"):
    """CIFAKE 폴더 구조가 버전마다 다를 수 있어, REAL/FAKE를 가진 'train' 디렉터리를 탐색."""
    for root, _dirs, _files in os.walk(base):
        if os.path.basename(root).lower() == "train":
            entries = set(os.listdir(root))
            if {"REAL", "FAKE"} <= entries:
                return root
    raise FileNotFoundError("CIFAKE train 폴더(REAL/FAKE)를 /kaggle/input 아래에서 못 찾음")


def main():
    install_detector()

    from detector.training.run import run_training

    data_root = find_train_dir()
    print("data_root:", data_root)

    run_training(
        data_root=data_root,
        out_dir="/kaggle/working",   # Kaggle은 여기만 결과로 저장됨
        epochs=3,
        batch_size=64,
        lr=1e-4,
        freeze_backbone=False,
    )


if __name__ == "__main__":
    main()
