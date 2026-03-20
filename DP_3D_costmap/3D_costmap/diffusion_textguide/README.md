# Diffusion TextGuide — 텍스트 조건 경로 계획

**VLM(Vision Language Model) 기반 텍스트 instruction**으로 조건을 주어  
Slope+Height costmap 기반 **CoT 효율 경로**를 생성하는 **텍스트 가이드 Diffusion** 모듈입니다.

---

## 디렉토리 구조

```
diffusion_textguide/
├── data/
│   ├── raw/                # A*로 생성한 초기 .pt 파일들 (Map, Path, Weight)
│   ├── visuals/            # generate_visuals.py 출력 이미지 (Slope/Height+경로)
│   ├── processed/          # 학습용으로 가공된 통합 데이터 (Batch 단위 등)
│   └── metadata/           # VLM이 생성한 instruction JSON 파일들
├── scripts/
│   ├── generate_data.py    # 지형·A* 경로 생성 (SlopeCotGenerator 클래스 포함) → data/raw/
│   ├── generate_visuals.py # VLM 제출용 이미지 생성
│   └── fetch_instructions.py # GPT/Gemini API 호출용
├── model/
│   ├── denoiser.py         # Diffusion U-Net/Transformer 구조
│   ├── encoder.py          # Map CNN & CLIP Adapter
│   └── diffusion_utils.py  # Gaussian process, Sampler 등
├── train.py                # 학습 메인 루프
├── inference.py            # 실제 대화형 경로 생성 테스트
└── README.md
```

---

## 현재 상태

| 파일/디렉토리 | 설명 |
|-------------|------|
| **scripts/generate_data.py** | ✅ 구현됨 — SlopeCotGenerator + A*·경로 유틸 내장, terrain + A* → data/raw/ 개별 .pt |
| **scripts/generate_visuals.py** | ✅ 구현됨 — data/raw/*.pt → Slope/Height+경로 시각화 → data/visuals/ 저장 |
| **inference.py** | 비어 있음 (placeholder) |
| **trainer.py** | 비어 있음 (placeholder) |
| **data/**, **model/**, **train.py** | 미구현 (예정) |

VLM 기반 텍스트 instruction 생성 및 **학습·추론** 파이프라인 구축 예정입니다.

---

## 모듈별 역할

### 데이터 파이프라인 (`data/`)

| 디렉토리 | 역할 |
|---------|------|
| **raw/** | `scripts/generate_data.py`로 생성한 초기 데이터<br>- 지형 맵 (Slope, Height)<br>- A* 경로 (다양한 weight: 0.1~0.9)<br>- 각 샘플별 `.pt` 파일 |
| **visuals/** | `scripts/generate_visuals.py` 출력<br>- Slope/Height 맵 + 경로 오버레이 PNG<br>- VLM 제출용 이미지 |
| **processed/** | 학습용 통합 데이터셋<br>- 배치 단위 텐서 변환<br>- 정규화된 costmaps, paths<br>- 텍스트 instruction 토큰화 |
| **metadata/** | VLM이 생성한 instruction JSON<br>- `{map_id: instruction_text}`<br>- GPT/Gemini API 응답 저장 |

### 스크립트 (`scripts/`)

| 파일 | 역할 |
|------|------|
| **generate_data.py** | 지형 생성 + A* 경로 탐색<br>- **SlopeCotGenerator** 클래스 및 A*·경로 유틸 내장<br>- 다양한 weight로 경로 생성 → `data/raw/` 저장 |
| **generate_visuals.py** | VLM 제출용 시각화 이미지 생성<br>- `data/raw/*.pt` 로드 → Slope/Height+경로 시각화<br>- `data/visuals/*.png` 저장 |
| **fetch_instructions.py** | GPT/Gemini API 호출<br>- 시각화 이미지 → VLM → instruction 텍스트<br>- JSON 저장 → `data/metadata/` |

### 모델 (`model/`)

| 파일 | 역할 |
|------|------|
| **denoiser.py** | Diffusion U-Net/Transformer<br>- 노이즈 예측 네트워크<br>- Map + Text 조건부 denoising |
| **encoder.py** | Map CNN & CLIP Adapter<br>- Costmap → Visual features (CNN)<br>- Text instruction → CLIP embeddings<br>- Multi-modal fusion |
| **diffusion_utils.py** | Diffusion 유틸리티<br>- Gaussian forward/reverse process<br>- Sampler (DDPM, DDIM 등)<br>- Guidance (Classifier-free 등) |

### 메인 스크립트

| 파일 | 역할 |
|------|------|
| **train.py** | 학습 메인 루프<br>- `data/processed/` 데이터 로드<br>- `model/` 모듈 조합<br>- WandB 로깅, 체크포인트 저장 |
| **inference.py** | 대화형 경로 생성 테스트<br>- 사용자 텍스트 입력<br>- 지형 맵 로드<br>- 경로 생성 및 시각화 |

---

## diffusion_patch와의 관계

**`diffusion_patch`**는 기본 텍스트 조건 기능을 제공합니다:
- **데이터**: `text_labels`, `text_tokens`, `vocab` (weight 기반 라벨)
- **모델**: `TextEncoder` + `CrossAttention`
- **학습/추론**: 텍스트 조건 지원

**`diffusion_textguide`**는 **VLM 기반 instruction 생성**으로 확장:
1. **자동 instruction 생성**  
   - 지형+경로 시각화 → VLM(GPT/Gemini) → 자연어 instruction
   - Weight 기반 라벨 대신 **맥락 기반 설명** 생성
2. **CLIP 기반 텍스트 인코딩**  
   - `model/encoder.py`에서 CLIP adapter 사용
   - 사전 학습된 언어 모델 활용
3. **대화형 추론**  
   - 사용자 자연어 입력 → 경로 생성
   - "가장 빠른 경로로", "안전한 경로로" 등 자유 형식 명령

---

## 사용 방법 (구현 시)

### 1. 데이터 생성 파이프라인

```bash
# 1. 지형 및 A* 경로 생성
python scripts/generate_terrain.py --config configs/default_config.yaml
# → data/raw/ 에 .pt 파일들 저장

# 2. VLM 제출용 시각화 생성
python scripts/generate_visuals.py --data_dir data/raw/
# → 시각화 이미지 생성

# 3. VLM API 호출하여 instruction 생성
python scripts/fetch_instructions.py --api gpt-4 --images_dir visuals/
# → data/metadata/instructions.json 저장

# 4. 학습용 데이터 가공
python scripts/process_data.py --raw_dir data/raw/ --metadata data/metadata/
# → data/processed/dataset.pt 저장
```

### 2. 학습

```bash
python train.py --config configs/default_config.yaml --data_dir data/processed/
```

- `data/processed/`에서 통합 데이터셋 로드
- `model/denoiser.py` + `model/encoder.py` 조합
- WandB 로깅, 체크포인트 저장

### 3. 추론

```bash
python inference.py --checkpoint checkpoints/best_model.pt --terrain_map terrain.npy
```

- 대화형 모드: 사용자 텍스트 입력 → 경로 생성
- 예: `"가장 빠른 경로로 가주세요"` → Diffusion 경로 생성

---

## 공유 리소스 (diffusion_patch)

| 항목 | 경로/모듈 | 활용 방안 |
|------|-----------|----------|
| 설정 | `diffusion_patch/configs/default_config.yaml` | 기본 설정 재사용 |
| 지형 생성 로직 | `diffusion_patch/generate_data.SlopeCotGenerator` | `scripts/generate_terrain.py`에서 재사용 |
| A* 경로 탐색 | `diffusion_patch/generate_data.a_star_cot_search` | `scripts/generate_terrain.py`에서 재사용 |
| Diffusion 스케줄러 | `diffusion_patch/diffusion.DiffusionScheduler` | `model/diffusion_utils.py`에서 참고 또는 재사용 |
| 경로 유틸 | `diffusion_patch/path_utils` | 경로 정규화/역정규화 등 |

`diffusion_textguide`는 **VLM 기반 instruction 생성**과 **CLIP 인코딩**을 추가하여 확장합니다.

---

## 의존성 (예정)

- **기본**: `torch`, `numpy`, `scipy`, `matplotlib`, `PyYAML`
- **VLM API**: `openai` (GPT), `google-generativeai` (Gemini)
- **CLIP**: `clip-by-openai` 또는 `transformers` (CLIP 모델)
- **학습**: `wandb`, `tqdm`
- **데이터**: `PIL`, `json`

---

## 요약

- **목적**: **VLM 기반 텍스트 instruction**으로 조건을 주는 Diffusion 경로 계획
- **핵심 차별점**: 
  - Weight 기반 라벨(`"Quickly"`, `"Safe route"`) 대신 **VLM이 생성한 자연어 instruction** 사용
  - **CLIP 기반 텍스트 인코딩** (사전 학습된 언어 모델 활용)
  - **대화형 추론** 지원 (자유 형식 텍스트 입력)
- **현재 상태**: 
  - `inference.py`, `trainer.py`는 placeholder
  - `data/`, `scripts/`, `model/`, `train.py` 미생성
- **확장 방향**: 
  - `diffusion_patch`의 지형 생성/A* 로직 재사용
  - VLM API 연동으로 instruction 자동 생성
  - CLIP adapter로 텍스트 인코딩 강화
