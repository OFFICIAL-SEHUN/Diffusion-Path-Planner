# Diffusion Patch — 경사 지형 경로 계획

경사(Slope) + 높이(Height) 2채널 costmap을 조건으로 **CoT(Cost of Transport) 효율 경로**를 생성하는 Diffusion 기반 경로 계획 모듈입니다.  
A* GT 경로와 비교·평가하고, **패치 기반 추론**을 지원합니다.

---

## 디렉토리 구조

```
diffusion_patch/
├── config_loader.py      # YAML 설정 로드
├── configs/
│   ├── default_config.yaml   # 기본 설정 (100×100, horizon 120 등)
│   └── bigger_config.yaml    # 대형 맵 (500×500, horizon 200)
├── cost_calculator.py    # 경로 CoT 비용 계산
├── data_loader.py        # Slope+Height 데이터셋 로더
├── diffusion.py          # DDPM 스케줄러 (forward/sample)
├── generate_data.py      # 학습용 terrain·경로 데이터 생성
├── loss.py               # DDPM MSE 손실
├── main.py               # Diffusion vs A* 비교·시각화 진입점
├── model.py              # ConditionalPathModel (U-Net + visual encoder)
├── patch_inference.py    # 패치 기반 Diffusion 추론
├── path_utils.py         # 경로 정규화/역정규화, PCA 선형성 검사
├── trainer.py            # 학습 루프 (WandB, 체크포인트)
├── visualization.py      # 경로 비교 시각화 (Diffusion / A* / Patch)
├── visualize_map.py      # terrain 맵·가중치별 경로 시각화
├── visualization/
│   └── visualize_gradient_data.py  # Slope+CoT 데이터셋 시각화
├── data/                 # 생성된 데이터 저장 (generate_data.py 실행 후)
│   └── test_dataset.pt
└── checkpoints/          # 학습 체크포인트 (trainer)
```

---

## 핵심 모듈 요약

| 파일 | 역할 |
|------|------|
| **config_loader** | `configs/*.yaml` 로드 |
| **configs/** | `data`(img_size, horizon), `gradient`(height_range, limit_angle 등), `diffusion`, `model`, `training` 설정 |
| **cost_calculator** | 경로 픽셀 + height map → **방향성 CoT** 계산, 구간별 오르막/내리막 통계 |
| **data_loader** | `data/test_dataset.pt`에서 costmaps(Slope, Height), paths, text_tokens 로드 → `GradientDataset` |
| **diffusion** | `DiffusionScheduler`: forward diffusion, `sample()` (조건부 역과정) |
| **generate_data** | `SlopeCotGenerator`로 terrain 생성, A* CoT 경로, 텍스트 라벨 → `test_dataset.pt` 저장 |
| **loss** | `StandardDiffusionLoss`: DDPM MSE(ε, ε_θ) |
| **main** | **진입점**: Diffusion vs A* 비교, CoT 계산, (옵션) 패치 추론, 시각화 |
| **model** | `ConditionalPathModel`: 1D U-Net + visual encoder, Slope+Height 조건, start/goal 조건 |
| **patch_inference** | costmap을 N×N 패치로 나누어 **패치별 Diffusion** 추론 후 경로 스플라인 결합 |
| **path_utils** | `denormalize_path`, `check_path_linearity` (PCA 기반 직선성 검사) |
| **trainer** | `GradientDataset` + `ConditionalPathModel` + `DiffusionScheduler`로 학습, WandB, 체크포인트 |
| **visualization** | `visualize_comparison`: Slope/Height 맵 위에 Diffusion, A*, Patch 경로 + CoT 요약 |
| **visualize_map** | terrain 맵과 weight별(0.1~0.9) A* 경로 시각화 |
| **visualization/visualize_gradient_data** | `test_dataset.pt` 내용(맵, 경로, CoT 등) 시각화 |

---

## 사용 방법

### 1. 데이터 생성

```bash
cd diffusion_patch
python generate_data.py --config configs/default_config.yaml
```

- 출력: `data/test_dataset.pt` (costmaps, paths, height_maps, slope_maps, text_tokens 등)
- `bigger_config.yaml` 사용 시 500×500 등 더 큰 맵으로 생성 가능

### 2. 학습 (Trainer)

학습 진입점은 `main.py`가 아닌 **`trainer` + `data_loader` + `model`** 조합입니다.  
`diffusion` 루트의 `main.py`처럼 `--mode train`을 쓰는 래퍼가 없다면, 직접 스크립트에서 예시처럼 구성하면 됩니다.

```python
from data_loader import GradientDataset
from model import ConditionalPathModel
from diffusion import DiffusionScheduler
from trainer import Trainer
from config_loader import load_config

config = load_config("configs/default_config.yaml")
dataset = GradientDataset(config, load_auxiliary=False)
model = ConditionalPathModel(config=config)
# DiffusionScheduler, Trainer 초기화 후 trainer.train()
```

체크포인트는 `config['training']['checkpoint_dir']`(기본 `checkpoints/`)에 저장됩니다.

### 3. Diffusion vs A* 비교 (메인 실행)

```bash
python main.py --config configs/default_config.yaml
```

- 지형 생성 → A* CoT 경로 탐색 → Diffusion 경로 생성 → **CoT 계산 및 비교** → 시각화

**옵션 예시:**

```bash
# 패치 기반 추론 사용 (4×4 패치, 배치 16)
python main.py --config configs/default_config.yaml --use-patches --num-patches 4 --batch-size 16

# 시작/끝 최소 거리 조정 (min_distance = img_size / factor)
python main.py --config configs/default_config.yaml --min-distance-factor 0.7
```

### 4. 시각화

- **경로 비교**: `main.py` 실행 시 `visualization.visualize_comparison`이 자동 호출됩니다.
- **Terrain 맵·가중치별 경로**:
  ```bash
  python visualize_map.py --config configs/default_config.yaml
  ```
- **데이터셋 시각화**:
  ```bash
  python visualization/visualize_gradient_data.py --data_path data/test_dataset.pt
  ```

---

## 설정 (config) 개요

- **data**: `num_samples`, `paths_per_terrain`, `img_size`, `horizon`, `min_distance_factor` 등
- **gradient**: `height_range`, `terrain_scales`, `mass`, `gravity`, `limit_angle_deg`, `pixel_resolution`, `pca_linearity_threshold` 등
- **diffusion**: `timesteps`, `beta_start`, `beta_end`
- **model**: `base_dim`, `time_embed_dim`, `image_feat_dim`
- **training**: `epochs`, `batch_size`, `learning_rate`, `checkpoint_dir`, `model_name`, `log_interval`

---

## 의존성

- `torch`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `PyYAML`
- 학습 시 `wandb` (Trainer)
- `generate_data` / `cost_calculator`: `path_utils`, `generate_data` 내부 CoT/경로 유틸

---

## 요약

- **입력**: 2채널 costmap (Slope, Height) + start/goal (및 선택적으로 텍스트)
- **출력**: CoT 효율 경로 (waypoints)
- **평가**: A* CoT 경로와 CoT·추론 시간 비교, 패치 기반 추론 지원
- **학습**: `GradientDataset` + `ConditionalPathModel` + `DiffusionScheduler` + `Trainer`로 DDPM 학습
