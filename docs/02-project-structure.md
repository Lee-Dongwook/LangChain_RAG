# 02 · 프로젝트 구조 — "연구 리포 → 서비스" 전환을 대비한 설계

> 지금은 모델링 연구가 목적이지만, 나중에 서비스화한다. 이 문서는 **나중에 갈아엎지 않을 폴더 구조**와 그 이유를 남긴다.

## 핵심 원칙: 재사용되는 코어를 분리하라

서비스가 실제로 쓰는 건 학습 코드 전체가 아니라 이 좁은 경로다:

```
이미지 입력 → 전처리 → 모델 로드 → 추론 → 확률/라벨 반환
```

이 경로에 쓰이는 **모델 정의 + 전처리(transform) + 추론 함수**는 학습에서도, 서비스에서도 쓴다.
→ 이 셋을 **코어 패키지 `detector/` 한 곳에 두고, 학습 코드와 서비스 코드가 둘 다 import** 하게 만든다.

## 왜 중요한가 — Train/Serve Skew

가장 악명 높은 배포 버그: **학습 때의 전처리와 서비스의 전처리가 다르면** 모델이 헛것을 본다.
학습 정확도는 95%인데 실서비스에선 60%로 떨어지고, 원인을 못 찾아 며칠 헤맨다.

**방지책:** 전처리(transform) 정의를 `detector/transforms.py` 한 파일에 두고 학습·추론이 같은 걸 import.
복붙 두 벌을 만들면 언젠가 반드시 어긋난다.

## 구조

```
detector/             # ★재사용 코어 패키지 (서비스가 import할 부분)
  __init__.py
  models/             # 모델 아키텍처 정의 (백본+헤드) — state_dict 로드하려면 필요
  transforms.py       # 전처리 정의 ★학습·추론 공유★
  inference.py        # predict(image) -> (label, prob)  ← 서비스가 호출할 함수
  data/               # Dataset, DataLoader (학습 전용)
  training/           # 학습 루프, 손실 (학습 전용)
  eval/               # 지표 (학습 전용)
  utils/              # 시드, 체크포인트 IO

scripts/              # 얇은 CLI 진입점 — 코어를 호출만 함
  train.py            #   → detector.training 호출
  eval.py
  predict.py          #   → detector.inference 호출

configs/              # 실험 설정 (yaml)
data/                 # 데이터셋 (gitignore)
experiments/          # 체크포인트 (gitignore)
notebooks/            # 탐색용
tests/

# 나중에 서비스화할 때 추가 (지금은 안 만듦):
api/                  # FastAPI 앱 — detector.inference를 import만 함
```

## 이 구조가 주는 것

- **지금**: `detector/` 안에서 연구. `scripts/train.py`는 얇은 껍데기(argparse + 호출)라 로직이 패키지에 쌓인다.
- **나중**: 서비스 만들 때 `api/`만 새로 만들고 `from detector.inference import predict` 한 줄. 학습 코드는 손 안 댐 → **코어 재사용, 재작성 0.**

## 지금 당장 지킬 실천 3개

1. **`transforms.py`를 따로 둔다** — Dataset 안에 전처리를 하드코딩하지 말고, 여기서 정의한 걸 주입받는다.
2. **모델 정의를 `models/`에 둔다** — 체크포인트(`state_dict`)를 로드하려면 모델 클래스를 다시 만들 수 있어야 하므로 학습·추론이 같은 정의를 import.
3. **`scripts/`는 얇게** — 진짜 로직은 `detector/` 안에. 스크립트는 "설정 읽고 → 패키지 함수 호출"만.

## 지금 신경 끌 것 (over-engineering 금지)

- FastAPI, Docker, 인증, DB — 전부 나중. 지금은 폴더만 안 만들면 됨. 구조만 잡아두면 `api/` 추가로 끝.
- 파일 몇 개는 처음엔 거의 비어 있어도 된다. **위치만 맞으면 OK.**

## 첫 코드 위치

- `Dataset` 클래스 → `detector/data/`
- 전처리 → `detector/transforms.py`에서 정의해 Dataset에 주입
