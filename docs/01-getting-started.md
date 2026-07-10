# 01 · 시작하기 — 초안 구축 로드맵과 선행 개념

> AI vs 사람 이미지 판별기 프로젝트의 **첫 초안을 어떤 순서로 세우고, 무슨 개념을 알고 시작해야 하는지** 정리한 학습 가이드.
> 코드는 개발자 본인이 직접 구현한다([../claude/CLAUDE.md](../claude/CLAUDE.md) 가이드 전용 모드 참고). 이 문서는 "무엇을·왜·어떤 순서로"의 지도다.

## 전체 그림 (파이프라인)

```
데이터 수집 → Dataset/DataLoader → 전처리·변환 → 모델(백본+헤드)
   → 순전파 → 손실 계산 → 역전파·옵티마이저 → 평가 → 체크포인트
```

각 단계마다 붙는 개념을 아래 순서대로 익힌다.

---

## 0단계 · 시작 전 반드시 잡고 갈 개념 (수학·PyTorch 기초)

이게 안 잡히면 뒤가 모래성이 된다.

- **Tensor**: shape, dtype, device(CPU/GPU), 이미지 배치 레이아웃 `(N, C, H, W)`. 이 4차원이 뭘 의미하는지 손에 익히기.
- **Autograd**: `requires_grad`, 계산 그래프, `.backward()`가 실제로 하는 일. "손실에서 파라미터까지 미분이 흘러간다"는 그림.
- **경사하강법**: gradient가 왜 파라미터 업데이트 방향인지, learning rate가 뭔지.
- **분류 기초**: logit → sigmoid/softmax → 확률, 그리고 **cross-entropy** 가 왜 분류 손실인지. 이진분류라 **BCE(Binary Cross Entropy)** 가 핵심.

> 이 4개는 "왜"를 스스로 설명할 수 있어야 다음이 의미를 가진다.

---

## 1단계 · 데이터 (모델보다 먼저!)

이 프로젝트는 **데이터가 8할**. 초보가 가장 과소평가하는 곳이며 도메인 함정이 여기 몰려 있다.

- **데이터셋 구성**: "사람 이미지" / "AI 이미지" 폴더, 라벨 0/1.
- **Train/Val/Test 분할**: 왜 나누는지 — val은 하이퍼파라미터 튜닝·조기종료 판단용, test는 끝까지 안 건드림.
- **데이터 누수(leakage)**: 같은 원본이 train·test에 동시에 들어가면 안 됨.
- ⚠️ **포맷·해상도 편향**: "AI는 전부 1024² PNG, 사람은 전부 JPEG" 같은 차이를 모델이 판별 대신 학습해버림. 데이터 쌍의 배포 조건을 맞추는 게 라벨만큼 중요.

**첫 초안**: 작은 데이터셋부터(사람/AI 각 수백~수천). 처음부터 수십만 장 욕심내지 말 것. 파이프라인이 도는지 확인이 먼저.

> 데이터 소스 선정은 큰 결정 — 진행 전에 논의(TODO).

---

## 2단계 · 입력 파이프라인 (Dataset / DataLoader / Transform)

- **`Dataset`**: 인덱스 i → `(이미지 텐서, 라벨)` 하나. 핵심은 `__len__`, `__getitem__`.
- **`DataLoader`**: 배치 묶기, 셔플, 병렬 로딩. `batch_size` · `shuffle` · `num_workers`.
- **Transform**: PIL → 텐서, 리사이즈, 정규화(Normalize).
  - ⚠️ **리사이즈·JPEG 재압축이 AI 아티팩트를 지운다.** 첫 초안은 단순하게 가되 "이 변환이 판별 근거를 훼손할 수 있다"는 걸 인지만 하고 넘어감.
- **Normalize**: 왜 ImageNet 평균/표준편차로 정규화하나 — 사전학습 백본이 그 분포로 학습됐기 때문.

---

## 3단계 · 모델 (파인튜닝의 핵심)

- **전이학습 / 파인튜닝**: ImageNet으로 잘 학습된 백본(예: ResNet50)의 특징 추출 능력을 빌리고, **마지막 분류 헤드만 우리 문제(2-class)로 교체**.
- **백본 + 헤드 구조**: torchvision ResNet의 마지막 1000-class fc를 2-class(또는 1-logit)로 교체.
- **Freeze vs Fine-tune**: 백본을 얼리고 헤드만 학습 vs 전체를 낮은 lr로 학습. 첫 초안은 "헤드만" 또는 "전체 낮은 lr"로 시작.

> 백본 로드는 torchvision 허용. 단, **헤드 교체·학습 루프는 직접 구현**이 학습 포인트.

---

## 4단계 · 학습 루프 (직접 구현의 본체)

한 에폭의 뼈대(의사코드):

```
for each batch (images, labels):
    optimizer.zero_grad()             # 이전 gradient 초기화
    outputs = model(images)           # 순전파
    loss = criterion(outputs, labels) # BCE / CE
    loss.backward()                   # 역전파 (autograd)
    optimizer.step()                  # 파라미터 업데이트
```

알아야 할 개념:

- **`zero_grad`가 왜 필요한가** — gradient가 누적되기 때문.
- **train 모드 vs eval 모드** (`model.train()` / `model.eval()`) — Dropout·BatchNorm 동작이 달라짐.
- **`torch.no_grad()`** — 평가 시 gradient를 끄는 이유(메모리·속도).
- **옵티마이저**: SGD vs Adam, learning rate의 영향.
- **에폭 / 이터레이션 / 배치**의 관계.

---

## 5단계 · 평가와 실험 관리

- **지표**: accuracy만 보면 안 됨. **AUROC**, precision/recall, confusion matrix. accuracy가 불균형·편향에 약한 이유.
- **오버피팅 감지**: train loss는 내려가는데 val loss가 오르는 신호.
- **체크포인트**: 가중치 저장/로드(`state_dict`).
- **재현성**: 시드 고정.

---

## 첫 초안 실행 순서 (성능보다 "관통"이 먼저)

1. **파이프라인 관통**: 이미지 20~50장(사람/AI 반반)으로 Dataset → DataLoader → 사전학습 ResNet 헤드 교체 → 몇 배치 학습 → loss가 떨어지는지 확인. 목표는 정확도가 아니라 **에러 없이 한 바퀴 도는 것**.
2. **확장**: 데이터 늘리기, val 분할 추가, 지표 측정.
3. **심화**: 오버피팅·cross-generator 일반화 문제로 진입.

이 순서를 어기고 큰 데이터 + 복잡한 모델로 시작하면 어디서 터졌는지 못 찾는다.

---

## 결정 로그

- **계산 환경**: GPU 없음 (맥북/노트북 CPU). → 저해상도·소형 데이터셋으로 시작.
- **첫 데이터 소스**: **CIFAKE** (Kaggle / Hugging Face).
  - 진짜 = CIFAR-10 실제 사진 6만, 가짜 = Stable Diffusion 1.4 생성 6만. 전부 **32×32**, balanced.
  - `train/{REAL,FAKE}`, `test/{REAL,FAKE}` 구조로 이미 분할·라벨링됨.
  - 장점: CPU에서 학습 가능, 해상도 편향 통제됨.
  - 한계: 생성기가 SD 1.4 하나뿐(→ 2단계 일반화 때 GenImage 등으로 확장), 32×32라 주파수 아티팩트 거의 없음.

## 1단계 실행 체크리스트 (데이터)

- [ ] CIFAKE 다운로드 (Kaggle 또는 HF)
- [ ] `data/raw/` 아래 배치, 폴더 구조 확인
- [ ] 눈으로 검증: 클래스별 장수 세기, 샘플 열어보기, 크기·포맷 일관성 확인
- [ ] val 분할 계획 세우기 (train 일부를 떼서 validation, test는 끝까지 미사용)

## 다음에 정할 것 (논의 TODO)

- [ ] **첫 백본**: ResNet18(가볍게) vs ResNet50 — CPU라 더 가벼운 쪽 유력
- [ ] 0단계 개념(Tensor shape, autograd, BCE) 자가진단 완료 여부
