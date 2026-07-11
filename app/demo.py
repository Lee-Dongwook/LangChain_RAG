"""CIFAKE 판별 웹 데모 (Gradio).

브라우저에서 이미지를 드래그&드롭하면 REAL / FAKE(AI 생성)를 확률과 함께 보여준다.

실행:
    python app/demo.py
    → http://127.0.0.1:7860
"""
import os

import gradio as gr

from detector.inference import load_model, predict_pil

CKPT = os.environ.get("CKPT", "experiments/kaggle_output/best.pt")

model, device = load_model(CKPT)


def classify(image):
    """Gradio 콜백: PIL 이미지 → gr.Label이 기대하는 {클래스: 확률} 딕셔너리."""
    if image is None:
        return {}
    _, _, probs = predict_pil(model, image, device)
    return probs


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="이미지 업로드"),
    outputs=gr.Label(num_top_classes=2, label="판별 결과"),
    title="CIFAKE Detector — 진짜 vs AI 생성 이미지",
    description=f"ResNet18 fine-tune (test acc 0.975) · device: {device}",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
