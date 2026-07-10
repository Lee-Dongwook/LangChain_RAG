# 03 · Dataset 설계도

> `detector/data/`에 직접 구현할 PyTorch `Dataset`의 설계 가이드. 완성 코드가 아니라 "무엇을·왜"의 지도.

## 개념 한 줄

> **"인덱스 하나 주면 학습 샘플 하나 `(이미지 텐서, 라벨)`를 돌려주는 객체."**

`torch.utils.data.Dataset`을 상속받고 **메서드 3개**를 구현한다.

## 3개 메서드

```
class CifakeDataset(Dataset):
    def __init__(self, root, transform):
        # 준비 (딱 1번): REAL/·FAKE/ 폴더를 훑어 (파일경로, 라벨) '목록'을 만든다.
        #                transform을 받아 저장.
        # ⚠️ 이미지를 여기서 전부 로드하지 마라 — 경로만 모은다 (lazy loading)

    def __len__(self):
        # 샘플 총 개수 (= 목록 길이)

    def __getitem__(self, idx):
        # idx번째 샘플 하나를 실제로 만든다:
        #   경로에서 이미지 열기 → RGB 변환 → transform 적용 → (이미지텐서, 라벨) 반환
```

## 두 가지 핵심 원리

- **`__init__`은 경로만, `__getitem__`이 실제 로딩** (lazy loading) — CIFAKE 12만 장을 다 메모리에 올리면 터짐.
- **transform은 밖에서 주입** — Dataset 안에 전처리 하드코딩 금지. `detector/transforms.py`에서 정의한 걸 받아 씀 (train/serve skew 방지).

## 반드시 밟는 함정 ⚠️

- **`.DS_Store`**: 맥 숨김 파일이 목록에 껴서 크래시 → **확장자(.jpg)로 필터링**.
- **`.convert("RGB")`**: 흑백/RGBA 섞이면 채널 수 달라져 배치에서 터짐 → 열자마자 RGB 통일.
- **라벨 dtype**: 손실 함수와 엮임. `BCEWithLogitsLoss`→float, `CrossEntropyLoss`→long. 지금은 인지만.
- **재현성**: 폴더 훑는 순서가 OS마다 다름 → 목록을 **sorted**로 정렬.

## 설계 결정 (직접 정할 것)

- `__init__` 인자: 최소 `root`(예: `data/raw/train`) + `transform`
- 라벨 매핑: `REAL=0, FAKE=1` (정하고 일관되게)
- 지금 transform 최소값: 32×32라 리사이즈 불필요 → **PIL→텐서 변환만** 있어도 파이프라인은 돈다. 정규화는 나중에.

## 다 짜면 자가 테스트 (넘어가기 전 필수)

1. `len(dataset)` → train 총 장수와 맞나?
2. `img, label = dataset[0]` →
   - `img.shape` == `(3, 32, 32)` (C, H, W)?
   - `img.dtype`, `label` 값 확인
3. REAL 첫 장과 FAKE 첫 장이 **다른 라벨**로 나오나?
