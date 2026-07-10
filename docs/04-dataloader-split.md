# 04 · train/val 분할 & DataLoader 설계도

> Dataset 다음 조각. `detector/data/`에 직접 구현할 "분할 + 배치 로딩"의 설계 가이드. 완성 코드가 아니라 "무엇을·왜".

## 1. train/val 분할

**왜**: 학습 중 오버피팅을 감지할 창문(val)이 필요. test는 끝까지 안 건드린다.
CIFAKE는 `train/`·`test/`만 주므로 **train에서 일부를 떼어 val**을 만든다.

**방법 두 갈래**
- **A. `torch.utils.data.random_split`** — train Dataset을 넣으면 두 조각(`Subset`)으로 쪼갬. 가장 간단.
- **B. 파일 목록 직접 분할** — 인덱스를 나눠 각각 Dataset 생성. 손은 더 가지만 통제력 큼.

**챙길 개념 3개**
- **재현성**: 분할 시 **시드(generator) 고정**. 안 그러면 실행마다 val이 바뀌어 어제 val이 오늘 train에 섞임 → 평가 오염.
- **비율**: 보통 train:val = 8:2 또는 9:1.
- **밸런스(stratify)**: val에도 REAL/FAKE가 골고루. CIFAKE는 50:50 + 대용량이라 랜덤 분할로도 대충 맞지만, "val이 한쪽으로 쏠리지 않았나" 확인할 가치 있음.

## ⚠️ 미묘한 함정 — 분할과 transform

`transform.py`에 **train용(증강 O)** / **eval용(증강 X)** 두 개를 만들었다. 그런데:

- `random_split`으로 **한 Dataset을 쪼개면, 두 조각이 같은 transform을 공유**한다.
- 즉 val에도 train 증강(RandomFlip/Rotation)이 걸려버림 → **val은 증강 없이 봐야** 정직한 평가인데 오염됨.

**해결 방향(힌트)**: "하나의 Dataset을 쪼개는" 방식은 이 문제를 갖는다. "train용/eval용 Dataset을 각각 만들고 **같은 인덱스로** 나누는" 방식은 이를 피한다. 어느 쪽으로 갈지가 이번 설계의 핵심 결정.

## 2. DataLoader

**개념**: Dataset은 "한 장씩" 주는 애, DataLoader는 그걸 **배치로 묶고·셔플하고·병렬 로딩**하는 애.

**핵심 파라미터**
- `batch_size` — 한 번에 몇 장 (CPU니 32~64로 시작)
- `shuffle` — **train은 True**(매 에폭 순서 섞기), **val/test는 False**
- `num_workers` — 병렬 로딩 프로세스 수. ⚠️ 맥 CPU면 우선 **`0`**으로 시작. 0보다 크면 macOS 멀티프로세싱 때문에 `if __name__ == "__main__":` 가드 없이는 에러/저하 가능.

**자가 테스트**: 배치 하나 꺼내 shape 확인
- 이미지: `(batch_size, 3, 32, 32)` = `(N, C, H, W)`
- 라벨: `(batch_size,)`

이 `(N, C, H, W)`가 곧 모델에 먹일 입력 형태.

## 파일 위치

- Dataset 정의(`dataset.py`)와 "그걸 조립하는 코드(분할·DataLoader)"를 분리하는 감각. 조립 로직은 `detector/data/`(예: `loaders.py`)나 학습 스크립트에서. 정답은 없음.
