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
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"


def install_detector():
    # ① torch/torchvision을 GPU 아키텍처에 맞는 CUDA 빌드로 강제 재설치.
    #    (Kaggle 기본 torch가 배정된 GPU를 지원 안 해 "no kernel image" 에러가 나서)
    #    --force-reinstall 없으면 "이미 설치됨"으로 pip가 건너뛰어 안 고쳐짐.
    #    반드시 torch를 import하기 전에 실행되어야 함.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         "torch", "torchvision", "--index-url", TORCH_INDEX],
        check=True,
    )
    # ② 우리 detector 패키지 설치 (의존성 없음 → torch 안 건드림)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", REPO],
        check=True,
    )


def find_dir_with_classes(base="/kaggle/input", name="train"):
    """basename이 name이고 REAL/FAKE를 가진 디렉터리를 /kaggle/input 아래에서 탐색."""
    for root, _dirs, _files in os.walk(base):
        if os.path.basename(root).lower() == name.lower():
            if {"REAL", "FAKE"} <= set(os.listdir(root)):
                return root
    raise FileNotFoundError(f"REAL/FAKE를 가진 '{name}' 폴더를 /kaggle/input 아래에서 못 찾음")


def main():
    install_detector()

    from detector.training.run import run_training
    from detector.data.ood_split import split_ood, materialize

    # ① CIFAKE 학습 폴더
    cifake_train = find_dir_with_classes(name="train")
    print("cifake_train:", cifake_train)

    # ② OOD(art)를 로컬과 동일 seed로 분할 → train만 학습에 사용.
    #    holdout은 실체화하지 않아 학습에 절대 섞이지 않는다(로컬 평가 전용).
    art_data = find_dir_with_classes(name="Data")
    print("art_data:", art_data)
    art_train = materialize(
        split_ood(art_data, holdout_ratio=0.2, seed=42)["train"],
        "/kaggle/working/art_train",
    )

    # ③ CIFAKE-train + art_train 혼합 학습
    run_training(
        data_root=[cifake_train, art_train],
        out_dir="/kaggle/working",   # Kaggle은 여기만 결과로 저장됨
        epochs=3,
        batch_size=64,
        lr=1e-4,
        freeze_backbone=False,
    )


if __name__ == "__main__":
    main()
