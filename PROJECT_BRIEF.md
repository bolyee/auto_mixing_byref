# PROJECT BRIEF — DDSP Vocal Auto-Mix (재작성용 핸드오프)

> 이 문서는 **새 대화에 컨텍스트를 통째로 넘기기 위한 문서**다.
> 이 디렉토리(`auto_mix 복사본`)는 **전면 재작성용 사본**이다. 기존 코드를 다 갈아엎어도 된다.
> **원본 `/Users/leehyunjoong/auto_mix` 는 절대 건드리지 말 것** (venv 포함).

---

## 1. 프로젝트가 뭐냐

내 원본 보컬(raw vocal)을 **레퍼런스 보컬의 음색·다이내믹·공간감에 자동으로 맞춰주는** 웹 기반 오토믹싱 시스템.
미분 가능한 DSP(DDSP) 모듈 + 경사하강법으로 EQ/컴프/리버브 파라미터를 최적화한다.
학부 캡스톤 디자인 프로젝트.

- 백엔드: FastAPI (`main.py`), 오디오 엔진 (`ddsp.py`), 음원분리 (`separation.py`)
- 프론트: `static/index.html` + JS (차트로 결과 시각화)
- 입력: raw 보컬 + 레퍼런스(아카펠라 또는 풀곡) + (선택)반주
- 출력: 처리된 보컬 / 풀믹스 / 마스터링본 + 매칭 오차 지표 + 차트 데이터

---

## 2. 재작성의 목표 (사용자가 원하는 것)

**DDSP 논문 기법으로 리버브 처리를 제대로 다시 만드는 것.** 핵심 아이디어 3개:

1. **DDSP 리버브로 리버브를 "줬다 뺏었다" 한다** — 같은 리버브 파라미터 θ로 add(컨볼루션)와 remove(디리버브)를 둘 다 수행.
2. **레퍼런스에서 리버브를 뺀다** — 레퍼런스는 이미 리버브가 걸려 있음. 이걸 제거한 `ref_dry`를 만들어야 EQ/컴프 매칭이 공정해짐.
3. **엔드투엔드로 따라간다** — 현재는 EQ→컴프→리버브가 각각 별도 옵티마이저인 greedy 방식. 이걸 하나의 그래프 + 하나의 손실로 통합.

---

## 3. 현재 신호 흐름 (재작성 전)

```
레퍼런스 풀곡 ──(선택)Demucs 분리──> 레퍼런스 보컬 (= 매칭 타겟)

raw vocal
  → 1. Differentiable Parametric EQ      (미분가능, Adam 500 epoch)
  → 2. 1176-style FET Peak Compressor    (numpy, 이진탐색 — 미분 불가!)
  → 2.1 Auto Makeup Gain                 (= -평균 GR, 실측)
  → 2.5 Differentiable Stereo Reverb     (미분가능, Adam 100 epoch)
  → 3. Safety Limiter (tanh soft clip)
  → 처리된 보컬 WAV
      → (선택) 반주와 믹스 → 풀믹스
          → (선택) 마스터링(톤매칭 + LUFS + 브릭월 리미터)
```

---

## 4. 현재 코드 구조

### `ddsp.py` (1689줄) — 오디오 엔진 전부

**유틸**
| 함수 | 줄 | 역할 |
|---|---|---|
| `fft_convolve` | 9 | FFT 컨볼루션 |
| `interpolate_sorted` | 33 | 정렬된 분포를 목표 길이로 선형보간 (Wasserstein용) |
| `compute_env_db_pytorch` | 47 | 프레임 RMS 포락선(dB) |
| `compute_decay_slopes_pytorch` | 58 | dB 포락선의 하강 기울기 추출 (`> -50dB` 게이트) |
| `extract_active_segments` | 69 | 5초 블록으로 잘라 RMS 상위 3개를 **이어붙임** (15초 학습 구간) |
| `compute_crest_factor_pytorch` | 105 | 크레스트 팩터 (L10 norm으로 peak 근사) |
| `compute_rms_variance_pytorch` | 117 | RMS 포락선 분산 |
| `compute_lra_stft_pytorch` / `compute_lra_ebu` | 129 / 252 | EBU R128 LRA |
| `get_equal_loudness_weights` | 296 | 등청감 가중치 |

**DDSP 모듈 (nn.Module)**
| 클래스 | 줄 | 파라미터 | 비고 |
|---|---|---|---|
| `DifferentiableReverb` | 181 | `raw_rt60`(1개), `raw_wet`(1개) | 지수감쇠 화이트노이즈 RIR, 스테레오, FFT conv. **전대역 RT60 1개뿐** |
| `DifferentiableEQ` | 320 | 밴드별 gain (기본 31밴드) | 가우시안 밴드, STFT magnitude에 곱함 |
| `DifferentiableCompressor` | 371 | threshold/ratio 등 | **STFT 도메인, 미분가능** — 현재 파이프라인에선 미사용 |
| `CrestFactorShaper` | 447 | (학습 없음) | 분위수 매핑 기반 다이내믹 정형 |

**1176 컴프레서 (numpy, 미분 불가)**
`_peak_follower`(711), `_syllable_peaks_from_envelope`(731), `_syllable_peak_variance`(759),
`_apply_1176_compressor`(768), `_binary_search_threshold`(831), `match_compression_1176`(869)
→ 레퍼런스의 **음절 피크 표준편차**를 타겟으로, ratio를 2:1→4:1→8:1→20:1→∞ 순으로 올리며
각 단계에서 threshold를 이진탐색(50회). 목표 달성하는 최소 ratio 채택.

**메인 엔트리**
| 함수 | 줄 | 역할 |
|---|---|---|
| `match_eq` | 1028 | **핵심 파이프라인 전부.** EQ+컴프+리버브+리미터+저장 |
| `mix_with_instrumental` | 1465 | 반주와 믹스, LUFS 기반 레벨 밸런스 |
| `master_track` | 1604 | 마스터링(톤매칭 + LUFS + 브릭월 리미터) |

`match_eq` 내부 단계 위치:
- STEP 1 EQ 최적화: ~1100–1200
- STEP 2 컴프: ~1150–1205
- STEP 2.1 오토 메이크업 게인: 1206–1216
- STEP 2.5 리버브 최적화: **1218–1320** ← 재작성 핵심 구간
- STEP 3 리미터 + 저장: 1322–

### `main.py` (220줄) — FastAPI

단일 엔드포인트 `POST /api/match-eq` (multipart form).

**요청 필드**
```
raw_vocal: File          ref_vocal: File          instrumental: File (optional)
num_bands: int = 5       match_amount: float = 1.0    smoothness: float = 1.0
hpf_freq: float = 80.0   lpf_freq: float = 16000.0    match_volume: bool = False
comp_amount: float = 1.0 reverb_amount: float = 1.0   mode: str = "both"
vocal_gain_db: float=0.0 separate_ref: bool = False   master: bool = False
```
- 허용 확장자: `.wav .mp3 .flac .ogg .m4a`
- `hpf_freq <= 20` → `None`, `lpf_freq >= 20000` → `None` 으로 변환 후 전달
- 파일은 `data/uploads/`, 결과는 `data/outputs/`, 세션은 `uuid4`
- 에러 시 생성 파일 전부 정리 후 HTTP 500

**응답 스키마 (프론트가 의존 — 재작성해도 유지 권장)**
```jsonc
{
  "status": "success",
  "audio_urls": { "raw", "ref", "processed",
                  "ref_song"?, "instrumental"?, "full_mix"?, "full_mix_raw"?, "full_mix_mastered"? },
  "separated_ref": bool,
  "match_error": float,      // 0.65*tonal + 0.35*dynamics
  "tonal_error": float,      // mel 포락선 MAE (dB), 100Hz 미만 제외
  "dynamics_error": float,   // 음절 피크 편차 오차
  "reverb_error": float,
  "chart_data": { "frequencies", "raw_envelope", "ref_envelope", "proc_envelope",
                  "eq_curve_x", "eq_curve_y", "bands_x", "bands_y" },
  "compression_data": { "gain_reduction_max", "cf_raw", "cf_ref", "cf_proc", "cf_error",
                        "rms_var_raw", "rms_var_ref", "rms_var_proc", "rms_var_error",
                        "dynamics_error", "src_dynamic_range", "ref_dynamic_range",
                        "shaped_dynamic_range", "ratio" },
  "reverb_data": { "rt60", "wet", "loss", "error_db", "similarity", "active" },
  "gate_data": null,
  "fullmix_data": {...} | null,
  "mastering_data": {...} | null
}
```
기타 라우트: `GET /` → `static/index.html`, `/static` 및 `/data` 정적 마운트.

### `separation.py` (114줄) — Demucs 음원분리
`torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS` (별도 pip 패키지 불필요, 가중치 자동 다운로드).
10초 세그먼트 + `Fade` overlap-add. CPU 추론이라 곡당 1~3분.
`separate_vocals(song_path, vocals_out_path, inst_out_path=None)` → dict.

---

## 5. 환경 — **실측 확인된 사실**

| 항목 | 값 |
|---|---|
| 머신 | **Apple M1, arm64**, Darwin 25.5.0 |
| 프로젝트 venv 파이썬 | **3.9.6** (`venv/`) |
| venv 내 주요 패키지 | torch **2.8.0**, torchaudio 2.8.0, **numpy 2.0.2**, librosa 0.11.0, scipy 1.13.1, pyloudnorm 0.2.0, fastapi 0.128.8 |
| 시스템 python3 | 3.9.6 (`/usr/bin/python3`, torch 2.7.1) |
| 다른 파이썬 | `/opt/homebrew/bin/python3` = **3.14.6**, `/opt/homebrew/bin/python3.10` = **3.10.20** |

### ⚠️ 알려진 환경 함정 (반드시 읽을 것)

**이 복사본의 `venv/bin/pip` 는 망가져 있다.**
shebang이 `#!/Users/leehyunjoong/auto_mix/venv/bin/python3` — 즉 **원본 프로젝트 venv를 가리킨다.**
`venv/bin/pip install ...` 를 실행하면 **원본에 설치된다.**

- 임시 회피: `venv/bin/python -m pip install ...` (이건 복사본 prefix로 정상 해석됨 — 확인함)
- 권장: **복사본 venv를 새로 만들어라.** 어차피 전면 재작성이므로.
  ```bash
  cd "/Users/leehyunjoong/auto_mix 복사본"
  rm -rf venv
  /usr/bin/python3 -m venv venv          # 3.9  (또는 /opt/homebrew/bin/python3.10)
  venv/bin/python -m pip install --upgrade pip
  ```

`requirements.txt` 현재 내용:
```
fastapi>=0.100.0  uvicorn>=0.20.0  python-multipart>=0.0.6
torch>=2.0.0  torchaudio>=2.0.0  numpy>=1.22.0,<2.0.0
librosa>=0.10.0  soundfile>=0.12.0  pyloudnorm>=0.1.0
```

---

## 6. Google `ddsp` 라이브러리 설치 가능성 — **실측 조사 결과**

`import ddsp` (magenta/Google 공식, PyPI `ddsp` 3.7.0, 마지막 릴리즈 **2023-05-25**)를 쓰려 했으나,
**M1 macOS에서 정상 설치 불가**. 실제 확인한 근거:

| 의존성 | 실측 결과 |
|---|---|
| `tensorflow<=2.11` | PyPI의 `tensorflow==2.11.0` macOS 휠은 **`macosx_10_14_x86_64` 뿐, arm64 없음**. arm64는 `tensorflow-macos==2.11.0`(`macosx_12_0_arm64`, cp38/39/310)에만 존재하는데 ddsp가 그 이름을 선언하지 않음 |
| `tflite-support<=0.1` | 릴리즈 파일이 **`tflite-support-0.1.0a1.tar.gz` 소스 1개뿐, 휠 0개**. bazel 빌드 필요 → 사실상 불가 |
| `crepe<=0.0.12` | py3.10 venv에서 실제 dry-run 결과 빌드 실패: `ModuleNotFoundError: No module named 'pkg_resources'` (최신 setuptools와 비호환) |
| `numpy<1.24`, `librosa<=0.10`, `scipy<=1.10.1`, `protobuf<=3.20` | 현재 스택(numpy 2.0.2 / librosa 0.11 / scipy 1.13)과 전면 충돌 |

**결론:** `--no-deps`로 억지 설치는 가능하다(`ddsp.core / synths / effects / losses`는 crepe·tflite를 import하지 않음).
하지만 그러면 TF 2.11 + torch 2.8이 한 FastAPI 프로세스에 뜨고 numpy를 1.23으로 내려야 해서 librosa/scipy가 깨진다.
그리고 `ddsp.effects.Reverb`는 **학습가능 IR 벡터 + FFT conv**로, 현재 `ddsp.py`가 이미 하고 있는 것과 본질적으로 같다.

### PyTorch 대안 (검증됨)

| 패키지 | 상태 | 내용 |
|---|---|---|
| **`dasp-pytorch` 0.0.1** | ✅ py3.9 arm64 설치 성공 (순수 파이썬, deps: torch/numpy/scipy) | `noise_shaped_reverberation` = **12밴드 옥타브별 gain+decay 필터드-노이즈 리버브** (Steinmetz et al., WASPAA 2021 — DDSP 리버브의 정석 확장판). 그 외 `parametric_eq`, `compressor`, `distortion`, `stereo_widener`, `stereo_panner`, `gain` 전부 미분가능 |
| `torchcomp` / `torchlpc` | ❌ 빌드 실패 | `clang++: error: unsupported option '-fopenmp'` — macOS clang에 OpenMP 없음. 쓰려면 `brew install libomp` + 플래그 지정 필요 |
| `nara-wpe` | 미검증, PyPI 존재 | 비미분 WPE 디리버브. 베이스라인 비교용으로 유용 |

> 참고: 조사 중 `dasp-pytorch`가 실수로 **원본 venv에 설치됐다가 제거되어 원상복구 완료**됨. 원본은 현재 깨끗함.

---

## 7. 현재 구현의 알려진 결함 (재작성 시 반드시 고칠 것)

1. **레퍼런스가 wet 상태로 EQ/컴프 타겟이 됨** — 레퍼런스 리버브 테일이 mel 포락선을 밀어올려 raw 보컬의 EQ가 과보정된다. 컴프도 마찬가지로, 리버브가 다이내믹을 이미 눌러놔서 음절 피크 편차가 과소측정된다. **이게 가장 큰 구조적 오류.**

2. **`DifferentiableReverb`의 RT60이 전대역 1개** (`ddsp.py:192,205`) — 실제 방은 주파수별 감쇠가 다르다(고역이 빨리 죽음). 1개 파라미터로는 원리적으로 매칭 불가.

3. **`extract_active_segments`가 비연속 5초 블록 3개를 그냥 `torch.cat`** (`ddsp.py:69–102`) — 블록 경계에서 **인위적 attack/decay가 생성**되어 decay slope 분포를 오염시킨다. 리버브 학습에 직접적 악영향. 경계 크로스페이드나 마스킹 필요. **현재도 버그.**

4. **`compute_decay_slopes_pytorch`의 `-50dB` 게이트** (`ddsp.py:64`) — 리버브 테일 대부분이 -50dB 아래라 정작 필요한 late reverb 정보를 버리고 있다. Schroeder 역적분 EDC 방식으로 가면 자연히 해결.

5. **greedy 단계 최적화** — EQ와 리버브가 서로의 결과를 못 본다. 옵티마이저가 3개 따로.

6. **1176 컴프가 numpy/이진탐색 = 미분 불가** (`ddsp.py:869`) — 엔드투엔드의 가장 큰 블로커. 그래디언트가 여기서 끊긴다. 정작 미분가능한 `DifferentiableCompressor`(`ddsp.py:371`)는 코드에 있는데 파이프라인에서 안 쓰인다.

7. **`ddsp.py`가 1689줄 단일 파일** — 모듈/파이프라인/손실/유틸이 전부 섞임. 재작성 시 분리 권장.

---

## 8. 재작성 목표 설계

### 8.1 구조
```
ref_wet ──dereverb_θ──> ref_dry ──┐
                                   ├──> EQ/Comp 매칭 타겟 (드라이끼리 = 공정)
raw ──EQ_φ──> Comp_ψ ──> y_dry ───┘
                          │
                          └──Reverb_θ──> y_wet ──> [ref_wet과 리버브 통계 비교]
```
**θ는 ref에서 뺄 때와 내 보컬에 줄 때 완전히 동일한 파라미터.** 이게 "줬다 뺏다"의 핵심 —
θ가 레퍼런스 방(room)의 정체를 표현하게 된다.

### 8.2 리버브 모듈
`dasp_pytorch.functional.noise_shaped_reverberation` 사용 (12밴드 gain + 12밴드 decay + mix)
또는 직접 구현: 옥타브 밴드 마스크(rfft 도메인 고정 버퍼) × 대역별 지수감쇠 × 스테레오 노이즈,
+ 프리딜레이 + early reflection 탭. 학습 파라미터 ~25개 수준.
(통짜 IR 벡터는 파라미터 88200개 → 그래디언트 노이즈 심함. 밴드 파라미터화가 안정적.)

### 8.3 디리버브 (미분가능)
컨볼루션 역연산은 non-minimum-phase라 불가능. **파워 도메인 스펙트럼 감산**으로 구현:
```
P = |STFT|²
P_late = causal_depthwise_conv(P, w)      # w[k] = exp(-2·6.9078·k·hop / (sr·rt60_band))
P_dry  = clamp(P - α·P_late, min=floor·P) # floor는 musical noise 방지, 필수
mag_dry = sqrt(P_dry)                      # 위상은 원본 STFT 위상 재사용
```
같은 θ의 `rt60`을 그대로 쓴다. 전부 미분가능.

### 8.4 손실 (raw와 ref는 다른 연주 → 파형 정렬 불가, 통계 도메인으로)
```
L = w1·L_tone     # mel 포락선: melenv(y_dry) vs melenv(ref_dry)
  + w2·L_dyn      # RMS 분위수 곡선: quantile(y_dry) vs quantile(ref_dry)
  + w3·L_reverb   # 대역별 EDC(Schroeder) 기울기 분포, 정렬 Wasserstein: y_wet vs ref_wet
  + w4·L_cycle    # ‖ dereverb_θ(Reverb_θ(raw)) − raw ‖
  + w5·L_dry      # dryness prior: ref_dry의 late/early 에너지비 최소화
```
- `L_reverb`는 **반드시 대역별**로 쪼개야 밴드별 decay 파라미터가 identifiable하다.
- `L_cycle` 없으면 디리버브의 α/floor가 θ와 무관하게 손실만 낮추는 방향으로 도망간다.
- `L_dry` 없으면 **θ→0 (리버브 없음)이 trivial solution**이다. 이 항이 "ref에서 리버브를 최대한 빼라"는 압력.

### 8.5 엔드투엔드
Adam 하나로 `{φ(EQ), ψ(Comp), θ(Reverb)}` 동시 최적화.
1176 컴프가 블로커이므로 선택:
- **(C) 컴프 고정, EQ+리버브만 조인트** — 제일 싸고 이득 대부분 나옴. 여기서 시작 권장.
- **(A) 조인트 루프에선 `dasp_pytorch.functional.compressor` 또는 `DifferentiableCompressor` 사용, 최종 렌더만 1176** — 추천 종착점.
- (B) 1176을 torch로 재작성 — 샘플단위 recursion이 느림, `torch.jit` 필요.

### 8.6 롤아웃 순서 (검증 가능하게)
0. **베이스라인**: `nara-wpe`로 ref만 디리버브 → 기존 파이프라인 그대로. EQ 매칭 개선폭 측정. 여기서 개선 안 보이면 나머지 재고.
1. 밴드별 리버브 교체 + 대역별 EDC 손실.
2. `dereverb_θ` 추가, `ref_dry`를 EQ/컴프 타겟으로. `L_dry` + `L_cycle`.
3. EQ+리버브 조인트 옵티마이저 통합 (안 C).
4. 미분가능 컴프까지 편입 (안 A).

### 8.7 함정
- **Demucs 분리 아티팩트**: `separate_ref=True`면 ref에 이미 스펙트럼 홀이 있다. 디리버브가 그걸 리버브로 오인함. `floor` 높게(0.1~0.15), 4kHz 이상은 디리버브 약하게.
- CPU 전용(GPU 없음). 현재 목표 응답시간 1~2초. Demucs 켜면 곡당 1~3분.
- 리버브 학습 구간은 15초로 캡. 전체 곡 컨볼루션은 마지막 1회만.

---

## 9. 디렉토리 현황

```
auto_mix 복사본/
├── ddsp.py                    1689줄 — 오디오 엔진 전부 (재작성 대상)
├── main.py                     220줄 — FastAPI
├── separation.py               114줄 — Demucs 보컬 추출
├── test_integration.py                — 통합 테스트
├── requirements.txt
├── Dockerfile, .dockerignore, run.sh
├── README.md                          — 상세 한국어 문서 (33KB, 현 구현 기준)
├── presentation_slides.md             — 발표자료
├── ddsp_paper.pdf                     — DDSP 논문 원문 (Engel et al., ICLR 2020)
├── static/                            — 프론트엔드 (index.html + JS)
├── data/                              — uploads/, outputs/
├── venv/                              — ⚠️ pip shebang 깨짐. 재생성 권장
└── *.wav, *.mp3                       — 테스트용 오디오 샘플
    ├── 38_LeadVox1.wav                  (16MB, 리드보컬 원본)
    ├── mixpractice inst.mp3             (반주)
    ├── mixpractice vocal my mix.mp3     (내 믹스)
    └── mixpractice0728.mp3
```

---

## 10. 참고 문헌

- **DDSP**: Engel et al., *"DDSP: Differentiable Digital Signal Processing"*, ICLR 2020. (`ddsp_paper.pdf`)
  - 리버브: 학습가능 IR 벡터 + FFT conv, `h[0]=0`으로 dry/wet 강제 분리
  - 멀티스케일 스펙트럴 손실: FFT 크기 (2048,1024,512,256,128,64), magnitude L1 + log-magnitude L1
  - 필터드 노이즈 신디사이저 (시변 FIR로 노이즈 성형)
- **Steinmetz, Ithapu, Calamia**, *"Filtered noise shaping for time domain room impulse response estimation from reverberant speech"*, WASPAA 2021. (`dasp_pytorch`의 `noise_shaped_reverberation` 근거)
- **Moorer**, *"About this reverberation business"*, CMJ 1979. (RIR = 직접음 + 초기반사 + 감쇠 노이즈 테일)
- ITU-R BS.1770 / EBU R128 (LUFS, LRA — 현재 코드에 구현되어 있음)
