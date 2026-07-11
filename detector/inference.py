import torch
import torch.nn.functional as F
from PIL import Image

from detector.models.resnet_finetune import build_resnet18
from detector.transform import get_eval_transform_resnet
from detector.training.run import pick_device

IDX_TO_LABEL = {0: "REAL", 1: "FAKE"}


def load_model(ckpt_path: str, device=None):
    """best.pt(state_dict)를 ResNet18 구조에 얹어 eval 모드로 반환."""
    device = device or pick_device()
    model = build_resnet18(num_classes=2, pretrained=False, freeze_backbone=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, device


@torch.no_grad()
def predict_pil(model, image: Image.Image, device, transform=None):
    """PIL 이미지 → (label, confidence, {클래스: 확률}).

    업로드(API/데모)처럼 파일 경로 없이 메모리에 있는 이미지를 판별할 때 쓴다.
    """
    transform = transform or get_eval_transform_resnet()
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)   # (1, 3, 128, 128)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)                   # (2,)
    idx = int(probs.argmax())

    return (
        IDX_TO_LABEL[idx],
        float(probs[idx]),
        {IDX_TO_LABEL[i]: float(probs[i]) for i in range(2)},
    )


def predict_image(model, image_path: str, device, transform=None):
    """단일 이미지 파일 경로 → (label, confidence, {클래스: 확률})."""
    return predict_pil(model, Image.open(image_path), device, transform)
