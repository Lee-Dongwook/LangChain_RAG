# AI vs Human Image Detector

이미지(추후 영상)가 **AI 생성물인지 사람이 만든 것인지** 판별하는 모델을 직접 모델링·파인튜닝하는 연구 리포.

- **1단계:** 이미지 이진 분류 (AI / 사람)
- **스택:** 순정 PyTorch + torchvision (고수준 래퍼 지양)
- **데이터:** [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (real=CIFAR-10, fake=Stable Diffusion 1.4, 32×32)
- 프로젝트 방향·규칙은 [claude/CLAUDE.md](./claude/CLAUDE.md), 학습 가이드는 [docs/](./docs) 참고.

## 실험 기록

데이터셋: CIFAKE (train 10만 → 8:2 train/val 분할). 지표: **val accuracy**.

| # | 모델 | 설정 | val acc | 메모 |
|---|---|---|---|---|
| 1 | SimpleCNN (직접 설계) | 32px, 전체 학습, 5ep | 0.94 | 작지만 태스크에 맞게 전 층 학습. 언더핏 여지 있었음 |
| 2 | ResNet18 (백본 freeze) | 128px, fc만 학습, 3ep | 0.87 | ImageNet 특징이 저수준 아티팩트 태스크엔 안 맞음 → 오히려 하락 |
| 3 | **ResNet18 (전체 파인튜닝)** | 128px, lr=1e-4, 3ep | **0.97** | 사전학습 시작점 + 태스크 맞춤 재조정. 현재 최고 |

**핵심 교훈:** 사전학습 특징이 새 문제와 안 맞으면 **얼리지 말고 풀어서 파인튜닝**해야 한다.
(freeze 0.87 < 밑바닥 SimpleCNN 0.94 < 전체 파인튜닝 0.97)

**정식 모델:** 3번(ResNet18 전체 파인튜닝, 캐글 GPU 학습)을 `models/cifake_resnet18.pt`로 승격.
학습에 쓰지 않은 **held-out test 2만 장 기준 accuracy 0.975** (REAL 9697/303, FAKE 9807/193).
val 0.976 → test 0.975 로 과적합 없이 유지됨.

> ⚠️ 이 점수는 **CIFAKE(= SD 1.4)** 기준. 다른 생성기(Midjourney/DALL·E)로의 일반화는 별도 평가 필요(아래).

## 일반화 평가 (out-of-distribution)

학습 분포(CIFAKE = 32px CIFAR 사진 + SD 1.4) **밖**의 데이터에서 얼마나 버티는지 측정.

| 평가셋 | 생성기·도메인 | 장수 | accuracy | 결과 |
|---|---|---|---|---|
| CIFAKE test (in-dist) | SD 1.4 · CIFAR 사진 | 20k | **0.975** | 정상 |
| [real-vs-fake art][ood-art] (OOD) | 타 diffusion · 아트 | 21.6k | **0.499** | **붕괴** |

**진단:** OOD에서 모델이 이미지의 **98.5%를 무조건 FAKE로 찍음** → 판별력 사실상 0.
in-dist 0.975 → OOD 0.499 격차 = 현재 모델은 **"CIFAKE 전용 스페셜리스트"**이지 범용 탐지기가 아님.
(단, 이 평가셋은 *생성기*와 *도메인*이 동시에 바뀌어 주범 분리는 안 됨.)

**재현:** `python scripts/eval.py --data-root data/ood/art/Data`

### 개선: 혼합 재학습 (진행 중)

누수 방지를 위해 art를 `art_train`(80%)/`art_holdout`(20%, 학습 금지)로 분리한 뒤,
**CIFAKE-train + art_train**을 섞어 재학습한다(멀티-루트 로더). 평가는 art_holdout으로.

| 모델 | art_holdout acc | 비고 |
|---|---|---|
| 원본 (CIFAKE만) | 0.499 | FAKE로 붕괴 |
| 혼합 (subset 2k·1ep, 플럼빙 검증) | **0.730** | 붕괴 풀리고 균형 회복 — 방향 실증 |
| 혼합 (전체·GPU) | _TBD_ | Kaggle 재학습 예정 |

작은 subset만으로도 붕괴가 풀림 → **학습 분포에 도메인을 섞으면 일반화가 회복**됨을 확인.
```bash
python scripts/split_ood.py          # art → art_train / art_holdout (seed 고정)
python scripts/train_mixed.py --quick # 로컬 빠른 검증 (전체는 --quick 빼고 GPU 권장)
python scripts/eval.py --ckpt experiments/mixed/best.pt --data-root data/ood/art_holdout
```

[ood-art]: https://www.kaggle.com/datasets/doctorstrange420/real-and-fake-ai-generated-art-images-dataset

## 학습 실행

```bash
# 로컬 (맥 M2 / MPS)
python scripts/train.py

# Kaggle GPU (헤드리스)
#   main에 push → GitHub Actions → kaggle kernels push → GPU에서 학습
#   결과 회수:
kaggle kernels output leedongwook0713/cifake-detector-train -p experiments/kaggle_output
#   → best.pt를 정식 모델로 승격: models/cifake_resnet18.pt
```

학습 로직은 `detector/training/run.py`의 `run_training()` 하나로, 로컬·Kaggle이 **경로만 바꿔 공유**한다.

## 추론 · 서빙

정식 모델 `models/cifake_resnet18.pt`를 불러 이미지를 판별한다. (`--ckpt` / `CKPT` 로 교체 가능)

```bash
# 단일 이미지 / 폴더 판별 (CLI)
python scripts/predict.py 사진.png
python scripts/predict.py 이미지폴더/

# held-out test 세트 재평가 (accuracy + 혼동행렬)
python scripts/eval.py

# 웹 데모 (드래그&드롭 UI)          → http://127.0.0.1:7860
python app/demo.py

# REST API + Swagger 문서            → http://127.0.0.1:8000/docs
uvicorn app.api:app --reload
curl -F "file=@사진.png" http://127.0.0.1:8000/predict
```

서빙 의존성 설치: `pip install -e ".[serve]"` (fastapi·uvicorn·gradio 등).
추론 로직은 `detector/inference.py`에 모아 학습과 동일한 모델/전처리를 재사용한다.

## 다음 마일스톤

- [x] **2단계 일반화 평가:** OOD(타 생성기·아트)에서 0.499로 붕괴 확인 → 위 "일반화 평가" 참고
- [ ] **2-1 개선:** 다생성기·다도메인 혼합 재학습으로 OOD 격차 축소
- [ ] 3단계 강건성(JPEG 재압축·리사이즈)
- [ ] 추후: 영상(프레임 단위)
