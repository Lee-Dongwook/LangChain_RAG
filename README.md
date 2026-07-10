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

> ⚠️ 이 점수는 **CIFAKE(= SD 1.4)** 기준. 다른 생성기(Midjourney/DALL·E)로의 일반화는 별도 평가 필요(다음 마일스톤).

## 학습 실행

```bash
# 로컬 (맥 M2 / MPS)
python scripts/train.py

# Kaggle GPU (헤드리스)
#   main에 push → GitHub Actions → kaggle kernels push → GPU에서 학습
#   결과 회수:
kaggle kernels output leedongwook0713/cifake-detector-train -p ./out   # best.pt, metrics.json
```

학습 로직은 `detector/training/run.py`의 `run_training()` 하나로, 로컬·Kaggle이 **경로만 바꿔 공유**한다.

## 다음 마일스톤

- [ ] **2단계 일반화 평가:** 학습에 안 쓴 생성기 데이터(GenImage 등)로 test → 진짜 성능 확인
- [ ] 3단계 강건성(JPEG 재압축·리사이즈)
- [ ] 추후: 영상(프레임 단위)
