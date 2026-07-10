# CLAUDE.md

이 파일은 이 리포에서 작업하는 Claude Code(및 사람 기여자)를 위한 프로젝트 헌장이자 가이드다.

## 프로젝트 목표

사용자가 넣은 **이미지(1단계)** 및 **영상(추후)** 이 **AI 창작물인지 사람이 만든 것인지**를 판별하는 모델을 만든다.

핵심 원칙 — **"최대한 직접 모델링·파인튜닝한다."** / **개발자인 Lee-DongWook에게 개발 방법에 대한 가이드를 제공할 뿐, 실제로 코드 작업을 수행하지 않는다.**

- 남이 만든 판별기를 그대로 갖다 쓰는 게 목표가 아니다. 학습 루프·손실·옵티마이저·평가 지표를 **직접 구현**하며 원리를 이해하는 것이 목표다.
- 다만 무에서 백본을 학습시키는 건 비현실적이므로, **사전학습 백본은 파인튜닝**한다. 순정 학습 목적과 현실적 성능 사이의 균형이다.

## 스코프 & 단계

1. **1단계 (현재): 이미지 판별.** 단일 이미지 입력 → AI/사람 이진 분류. 여기에 집중한다.
2. **2단계: 일반화.** 학습에 안 쓴 생성기(예: SD로 학습 → Midjourney로 테스트)에 대한 cross-generator 성능. 이 프로젝트의 진짜 어려운 문제.
3. **3단계: 강건성.** JPEG 재압축·리사이즈·크롭 등 후처리에도 견디는지.
4. **추후: 영상.** 프레임 단위 추출 + 시간축 집계. 1단계 이미지 판별기를 프레임 분류기로 재사용하는 방향.

산출물 형태는 **모델링 연구 리포**다. 학습/평가 스크립트·노트북·실험 관리가 중심이며, 사용자 UI(Streamlit/API 등)는 아직 만들지 않는다.

## 기술 스택

- **언어:** Python 3.10+
- **핵심:** 순정 **PyTorch**. 학습 루프·손실·평가는 직접 작성한다.
- **허용:** `torchvision`(사전학습 백본 로드 + 이미지 변환), `numpy`, `pillow`, `matplotlib`, `scikit-learn`(지표), `pyyaml`(설정), `tqdm`.
- **지양(고수준 래퍼):** PyTorch Lightning, HuggingFace `Trainer`, `timm`, fastai. 파인튜닝·학습 루프를 이들이 대신 짜주면 학습 목적이 흐려진다. **꼭 필요해지면 그때 사용자와 상의 후 도입한다.**
- 실험 로깅은 우선 파일(JSON/CSV) + matplotlib로 시작. `wandb`/`tensorboard`는 필요해지면 논의 후 추가.

## 리포 구조 (목표)

아직 대부분 비어있다. 코드가 생기면 아래 구조를 따른다.

```
data/                 # 데이터셋 (gitignore — 커밋 금지)
  raw/                # 원본 다운로드
  processed/          # 전처리 산출물
src/
  data/               # Dataset, DataLoader, transform 정의
  models/             # 모델 아키텍처 (백본 래핑 + 커스텀 헤드/모듈)
  training/           # 학습 루프, 손실, 옵티마이저/스케줄러
  eval/               # 지표, cross-generator·강건성 평가
  utils/              # 시드 고정, 로깅, 체크포인트 IO
configs/              # 실험 설정 (yaml)
scripts/              # train.py, eval.py, predict.py 진입점
notebooks/            # 탐색·시각화 (재현 로직은 src로 승격)
experiments/          # 체크포인트·로그 (gitignore)
tests/
```

## 도메인 메모 (AI vs 사람 이미지 판별에서 중요한 것)

작업 시 아래를 염두에 둘 것. 이 프로젝트는 "그냥 CNN 이진분류"로 끝나지 않는다.

- **주파수 도메인 아티팩트:** GAN/디퓨전 생성물은 FFT/DCT 상에 사람 눈에 안 보이는 스펙트럼 지문을 남긴다. RGB 분류기에 주파수 특징을 결합하면 성능이 오르는 경우가 많다.
- **전처리 함정:** JPEG 압축·리사이즈·리샘플링이 바로 그 아티팩트를 지워버린다. 전처리 파이프라인이 판별 근거를 파괴하지 않는지 항상 의심할 것. 리사이즈 보간법·JPEG 품질을 실험 변수로 관리한다.
- **일반화가 핵심 난제:** 특정 생성기(SD 등)에만 오버핏하면 새 생성기에서 무너진다. **학습에 쓴 생성기와 평가에 쓴 생성기를 반드시 분리**해서 리포트한다. train/val/test는 생성기 단위로도 쪼갠다.
- **데이터 밸런스·누수:** "사람" 데이터와 "AI" 데이터의 해상도·포맷·출처가 다르면 모델이 판별이 아니라 그 편향을 학습한다(예: AI는 전부 PNG 1024², 사람은 전부 JPEG). 데이터 쌍의 배포 조건을 맞추는 게 라벨만큼 중요하다.
- **평가 지표:** 정확도만 보지 말 것. AUROC, 생성기별 정확도, 후처리 조건별 정확도를 함께 본다.

## 개발 규칙 (Claude가 이 리포에서 지킬 것)

- **재현성:** 모든 학습은 시드 고정. 실험은 `configs/`의 yaml로 하이퍼파라미터를 남기고, 하드코딩하지 않는다.
- **직접 구현 우선:** 학습 루프·손실·지표는 라이브러리 한 줄 호출로 대체하지 말고 `src/`에 명시적으로 작성한다(학습 목적). torchvision 백본 로드는 예외.
- **데이터·체크포인트는 커밋하지 않는다.** `data/`, `experiments/`, `*.pt`, `*.pth`는 gitignore 대상.
- **큰 결정은 상의:** 새 무거운 의존성 도입, 데이터셋 선택, 아키텍처 방향 전환은 진행 전에 사용자에게 확인한다.
- 노트북에서 검증된 로직은 `src/`로 승격해 재사용 가능하게 만든다.

## 명령어 (환경 구성 후 사용 예정 — 현재는 미구현)

```bash
# 환경
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 아직 없음: 첫 코드 작성 시 생성

# 학습 / 평가 / 추론 (진입점 생성 후)
python scripts/train.py   --config configs/baseline.yaml
python scripts/eval.py    --config configs/baseline.yaml --checkpoint experiments/<run>/best.pt
python scripts/predict.py --checkpoint experiments/<run>/best.pt --image path/to/img.png
```

## 현재 상태

- 리포는 사실상 비어있다(과거 LangChain RAG 템플릿 잔재를 정리한 상태).
- 다음 착수 지점: (1) `requirements.txt` + 프로젝트 골격 생성, (2) 데이터 소스 선정, (3) 사전학습 백본 파인튜닝 베이스라인(예: torchvision ResNet50 이진 분류) 구축.
