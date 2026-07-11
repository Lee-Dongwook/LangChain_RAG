import argparse
import os

from detector.inference import load_model, predict_image

IMG_EXTS = (".jpg", ".jpeg", ".png")


def _collect(path: str):
    if os.path.isdir(path):
        return [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith(IMG_EXTS)
        ]
    return [path]


def main():
    parser = argparse.ArgumentParser(description="CIFAKE REAL/FAKE 판별기")
    parser.add_argument("target", help="이미지 파일 또는 이미지가 든 폴더")
    parser.add_argument("--ckpt", default="experiments/kaggle_output/best.pt",
                        help="체크포인트 경로 (기본: 캐글 학습 결과)")
    args = parser.parse_args()

    model, device = load_model(args.ckpt)
    print(f"device: {device} | ckpt: {args.ckpt}\n")

    for img_path in _collect(args.target):
        label, conf, probs = predict_image(model, img_path, device)
        print(f"{os.path.basename(img_path):40s} → {label:4s} "
              f"({conf*100:5.1f}%)  REAL={probs['REAL']*100:4.1f}% FAKE={probs['FAKE']*100:4.1f}%")


if __name__ == "__main__":
    main()
