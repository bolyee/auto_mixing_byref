---
marp: true
theme: default
_class: lead
paginate: true
math: katex
backgroundColor: #ffffff
color: #1f2937
style: |
  section {
    font-family: 'Inter', 'Noto Sans KR', sans-serif;
    padding: 36px 50px;
    background-color: #ffffff;
    color: #1f2937;
    font-size: 23px;
  }
  h1 { color: #1e3a8a; font-size: 1.8em; margin-bottom: 0.3em; font-weight: 700; }
  h2 { color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px;
       font-size: 1.25em; margin-top: 0; font-weight: 600; }
  h3 { color: #2563eb; font-size: 1.0em; margin-bottom: 6px; font-weight: 600; }
  footer { color: #6b7280; font-size: 0.55em; }
  code { background-color: #f3f4f6; color: #1e40af; padding: 2px 6px; border-radius: 4px;
         font-family: 'Fira Code', 'Consolas', monospace; border: 1px solid #e5e7eb; }
  blockquote { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 10px 16px;
               margin: 10px 0; color: #1e40af; border-radius: 0 6px 6px 0; font-size: 0.9em; }
  ul, ol { margin-top: 6px; margin-bottom: 6px; font-size: 0.9em; line-height: 1.5; }
  li { margin-bottom: 4px; }
  strong { color: #1d4ed8; font-weight: 600; }
  table { font-size: 0.76em; width: 100%; border-collapse: collapse; margin: 10px 0; }
  th, td { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }
  th { background-color: #f8fafc; color: #1e3a8a; }

  /* 목차 전환 슬라이드 (TOC Transition) 스타일 */
  .toc-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 18px;
  }
  .toc-item {
    display: flex;
    align-items: center;
    padding: 10px 18px;
    border-radius: 8px;
    background-color: #f8fafc;
    border: 1.5px solid #e2e8f0;
    color: #94a3b8;
    font-size: 0.95em;
    font-weight: 500;
  }
  .toc-item .num {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 28px;
    height: 28px;
    border-radius: 6px;
    background-color: #e2e8f0;
    color: #64748b;
    font-weight: 700;
    font-size: 0.85em;
    margin-right: 14px;
    flex-shrink: 0;
  }
  .toc-item.active {
    background-color: #eff6ff;
    border: 2px solid #2563eb;
    color: #1e3a8a;
    font-weight: 700;
    box-shadow: 0 4px 10px -2px rgba(37, 99, 235, 0.18);
    transform: scale(1.015);
  }
  .toc-item.active .num {
    background-color: #2563eb;
    color: #ffffff;
  }
  .toc-item.active .title {
    color: #1e3a8a;
    font-weight: 700;
  }
---

# 레퍼런스 기반 AI 보컬 믹싱 & 마스터링 시스템

### 레퍼런스 가이드 역전파를 통한 자율 음향 최적화

**발표자: 이현중**
2026.09

---

## 전체 목차 (Overview)

1. **믹스의 정의 및 전체 시스템 아키텍처**

2. **핵심 도구 3종의 수학적 모델링 & 플러그인 설계**

3. **최적화 메커니즘 (Optimization Mechanism)**

4. **풀믹스 합산 (Summing with Instrumental)**

5. **마스터링 체인 (Mastering Chain)**

6. **결론 (Conclusion)**


---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item active"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 & 플러그인 설계</span></div>
  <div class="toc-item"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 1. 믹스(Mixing)란 무엇인가?

* **정의**: 녹음된 개별 트랙들(보컬, 드럼, 베이스, 신스 등)을 음향학적·음악적으로 조율하여 단 하나의 트랙으로 결합하는 엔지니어링 과정.
* **믹싱의 3대 차원 (3D Audio Space)**:
  1. **주파수 차원 (Frequency / Height)**: 각 악기가 담당하는 주파수 영역을 분할하여 마스킹(소리 묻힘) 방지(ex. 저역이 풍부한 반주와 보컬).
  2. **다이내믹스 차원 (Dynamics / Depth)**: 소리의 크고 작음의 폭을 제어하여 전후 거리감과 에너지 균일성 확보(ex. 하이라이트에서만 큰 보컬을 반주 속에 어떻게 넣느냐).
  3. **시간·공간 차원 (Time & Space / Width & Reverb)**: 잔향을 통해 가상의 입체 음향 공간 구축.

> **보컬 믹싱의 궁극적 목표**: 보컬이 반주(MR)에 묻히지 않고 명료하게 들리면서도, 하나의 공간 안에서 일체감 있게 어우러지도록 만드는 것.

---

## 2. 실제 믹싱 현장에서는 어떤 작업을 하는가?

실제 스튜디오에서 엔지니어가 보컬 트랙을 처리하는 4단계 핵심 워크플로우:

1. **톤 쉐이핑 (Tone Shaping - EQ사용)**:
   - 보컬을 분석하여 특정 주파수대역을 깎거나(Cut) 부스트(Boost)하여 목표 주파수 밸런스 형성.
2. **다이내믹스 제어 (Dynamic Control - Compressor 사용)**:
   - 작은 소리부터 큰 샤우팅까지의 음량 편차를 줄여, 반주 위에서 일정한 에너지를 유지하도록 압축.
3. **공간감 형성 (Spatialization - Reverb 사용)**:
   - 무향실(흡음 부스)에서 건조하게 녹음된 보컬에  잔향(Tail)을 부여.
4. **레벨 밸런싱 & 서밍 (Level Balancing & Summing)**:
   - 보컬과 반주의 음량을 정밀하게 정렬하여 합산(Mixdown).

---

## 3. 기존 연구의 한계 vs 본 연구의 결정적 차이점

### ⚠️ 기존 믹싱 연구의 한계:
* **드라이(Dry) 신호의 부재**: 상용 발매곡에서 가공 전 순수한 원본 보컬을 구하는 것은 불가능함.
* **환경과 재료의 불일치**: 설령 프로 엔지니어의 설정값을 알아낸다 해도, **"그 수치가 내 보컬에도 잘 어울릴까?"**라는 근본적 한계 존재.
* 🍳 **요리에 비유하자면**:
  - 아무리 유명 셰프의 정밀한 레시피가 있어도, **내 주방 환경과 내가 가진 식재료의 상태가 다르면 다른 요리가 됨**이 됨.
  - 즉, 남의 프리셋이나 정답 데이터셋을 단순 복제하는 방식은 실제 환경에서 작동하지 않음.

### 💡 본 연구의 접근: Reference-Guided Unsupervised Optimization
* **완성된 맛을 나침반 삼아 자율 조율**: 레시피(파라미터)를 맹목적으로 베끼는 대신, 닮아가고 싶은 **완성된 요리(레퍼런스 음원)의 풍미(음색·다이내믹스 포락선)**를 목표로 설정.
* **내 재료 맞춤형 역전파**: 대규모 정답 데이터셋 없이, 미분 가능한 DSP(DDSP)를 통해 **내 보컬의 특성에 꼭 맞는 최적의 믹싱값을 직접 최적화**.

---

## 4. 전체 시스템 아키텍처 및 데이터 흐름도

![width:1040px center](system_architecture.svg)

---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item active"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 &amp; 플러그인 설계</span></div>
  <div class="toc-item"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 5. 핵심 도구 3종의 역할과 공학적 목적

| 플러그인 | 음향학적 역할 | 공학적 목적 (최적화 목표) |
|---|---|---|
| **이퀄라이저<br>(Equalizer, EQ)** | 주파수 응답($H(f)$) 보정 | 레퍼런스 보컬의 **스펙트럼 포락선(Mel Spectral Envelope)**과 내 보컬의 스펙트럼 거리 최소화 |
| **컴프레서<br>(Compressor)** | 신호의 다이나믹레인지 감소 | 큰 신호를 눌러 **다이내믹 레인지(Crest Factor)**를 레퍼런스와 일치시킴 |
| **리버브<br>(Reverb)** | 실내 음향 임펄스 응답(RIR) 잔향 부여 | 반주 공간에 맞는 잔향을 생성하여 입체감 형성 |

---

## 6. 미분 가능 EQ 설계: 17-Band Constant-Q 필터뱅크

주파수 축을 인간 청각 특성(바크/멜 스케일)에 맞추어 **옥타브 단위 등간격(Constant-Q)**으로 분할:

* **중심 주파수 배치 ($J=17$, 범위 $200\text{ Hz} \sim 10\text{ kHz}$)**:
  $$f_j = 200 \cdot 50^{\frac{j}{J-1}} \quad (j = 0, 1, \dots, 16)$$
* **로그 주파수 간격 $\Delta$ 및 대역폭 $\sigma$**:
  $$c_j = \log_2 f_j, \qquad \Delta = \frac{\log_2 50}{16} \approx 0.3527\text{ oct}, \qquad \sigma = 0.6\Delta \approx 0.2116\text{ oct}$$
* **가우시안 필터뱅크 행렬 $\Phi \in \mathbb{R}^{F \times J}$ ($F=1025$ STFT bins)**:
  $$\Phi(f, j) = \exp\left( -\frac{(\log_2 f - c_j)^2}{2\sigma^2} \right)$$
* **전체 EQ 게인 커브 ($G_{\text{scale}} = 15\text{ dB}$, 학습 파라미터 $\theta_{\text{eq}} \in \mathbb{R}^{17}$)**:
  $$G_{\text{dB}}(f) = \sum_{j=0}^{J-1} \Phi(f, j) \cdot (G_{\text{scale}} \cdot \theta_j)$$

---

## 7. 왜 200 Hz – 10 kHz 대역만 제어하는가?

수학적 손실 함수가 음향 신호의 "물리적 결함"을 잘못 학습하는 현상을 방지하기 위한 핵심 필터링:

1. **하한선 200 Hz 제외 이유 (음원 분리 마스킹 손실)**:
   - 상용 레퍼런스 곡에서 보컬을 분리(Source Separation)할 때, $100 \sim 150\text{ Hz}$ 대역은 베이스/킥드럼에 심하게 마스킹되어 **보컬 정보 자체가 소실**됨.
   - 정보가 유실된 저역을 타깃으로 잡으면 EQ가 저역을 비정상적으로 왜곡 컷/부스트함.
2. **상한선 10 kHz 제외 이유 (코덱 절벽 현상)**:
   - 손실 압축 음원(MP3/AAC)은 고역($>15\text{ kHz}$)을 강제로 잘라내는 **코덱 컷오프**를 지님.
   - 실제 측정 시 17 kHz 부근에서 $\text{raw} - \text{ref}$ 오차가 **$+50\text{ dB}$**까지 치솟음.
   - 이를 학습하면 EQ가 초고역을 극단적으로 깎아버려 보컬의 공기감(Air)이 파괴됨.

> **설계 원칙**: 평가 대역 $\Omega$와 제어 대역을 $[200, 10000]\text{ Hz}$로 일치시켜 **신뢰할 수 있는 구간에서만 수학적 최적화** 수행.

---

## 8. EQ의 위상 보존 및 에너지 보존 메이크업

### 1) 위상 보존 주파수 영역 곱셈
EQ 처리는 STFT 영역에서 위상(Phase)을 보존한 채 진폭에만 실수 게인을 곱함:
$$\mathcal{G}(f) = 10^{\alpha \cdot G_{\text{dB}}(f) \cdot M_{\text{eq}}(f) / 20}, \qquad S'(f, t) = \mathcal{G}(f) \cdot S(f, t)$$
- 위상 왜곡(Phase Distortion)이 없어 빗살 필터(Comb Filtering) 현상이 원천 차단됨.

### 2) 에너지 보존 정규화 (Energy Conservation)
EQ로 특정 대역을 부스트하면 보컬 전체 볼륨이 커져 컴프레서의 동작점이 틀어짐:
$$g_{\text{norm}} = \sqrt{\frac{\sum_{f, t} |S(f, t)|^2}{\sum_{f, t} |S'(f, t)|^2 + \epsilon}}, \qquad y_{\text{eq}} = \text{iSTFT}(g_{\text{norm}} \cdot S')$$
- **효과**: 순수한 "톤(Tone)" 변화만 남기고 전체 음향 파워를 $1.0$으로 일정하게 보존.

---

## 9. 미분 가능 컴프레서: 피크 검출기와 전달함수

시간 영역에서 동작하며 autograd 역전파가 가능한 컴프레서 모델:

* **1단계: 블록 피크 검출기 ($H_d=128$, $W_d=256$, $2.9\text{ ms}$ 간격)**
  $$e_k = \max_{i \in [kH_d, kH_d+W_d)} |u[i]|, \qquad E_k = 20\log_{10}(e_k + \epsilon)$$
* **2단계: 1차 IIR 평활화 (Attack $\tau_a=0.5\text{ ms}$, Release $\tau_r=120\text{ ms}$)**
  $$\alpha_a = \exp\left(-\frac{H_d}{f_s \tau_a}\right), \quad \alpha_r = \exp\left(-\frac{H_d}{f_s \tau_r}\right)$$
  $$\bar{E}_k = \begin{cases} \alpha_a \bar{E}_{k-1} + (1-\alpha_a)E_k & (E_k > \bar{E}_{k-1}) \\ \alpha_r \bar{E}_{k-1} + (1-\alpha_r)E_k & (E_k \le \bar{E}_{k-1}) \end{cases}$$

---

* **3단계: 미분 가능한 Soft-Knee 전달함수 ($K=4\text{ dB}$, 고정 Ratio $R=10:1$)**
  $$F(y_k) = \begin{cases} y_k & (y_k \le T - K/2) \\ y_k + \left(\frac{1}{R}-1\right)\frac{(y_k - T + K/2)^2}{2K} & (|y_k - T| \le K/2) \\ T + \frac{y_k - T}{R} & (y_k > T + K/2) \end{cases}$$
* **4단계: 오토 메이크업 게인(Auto Makeup) 및 최종 출력 파형 계산 (`modules.py:392-408`)**
  $$\text{GR}_k = F(\bar{E}_k) - \bar{E}_k \le 0\text{ dB}, \qquad \overline{\text{GR}}_{\text{active}} = \frac{1}{|M_{\text{active}}|} \sum_{k \in M_{\text{active}}} \text{GR}_k$$
  $$\text{net\_gain}_k = \text{GR}_k - \overline{\text{GR}}_{\text{active}}, \qquad g_k = 10^{\text{net\_gain}_k / 20}$$
  $$\tilde{g}[n] = \text{Interp}(g_k), \qquad y_{\text{comp}}[n] = u[n] \cdot \tilde{g}[n]$$
  - EQ의 에너지 보존($g_{\text{norm}}$)처럼, 활성 발성 구간($M_{\text{active}}$)의 평균 감쇄량을 즉시 복원하여 사용 가능한 레벨을 유지 (역전파 손실이 아닌 DSP 순전파 물리 복원).

---

## 10. 미분 가능 리버브

실내 음향 잔향(Reverberation)을 물리적으로 모사하고 시간 영역에서 미분 가능한 리버브 모델:

* **1단계: 공간 지수 감쇄 포락선 (Decay Envelope, $\text{RT}_{60}$)**
  - 잔향 시간 $\text{RT}_{60}$(에너지가 $-60\text{ dB}$ 감쇄하는 시간)에 따른 감쇄율 $\alpha$:
    $$\alpha = \frac{3 \ln 10}{\text{RT}_{60} \cdot f_s} \approx \frac{6.9078}{\text{RT}_{60} \cdot f_s}, \qquad e(t) = \exp(-\alpha t) \quad (0 \le t < T_{\text{dur}})$$
* **2단계: 합성 RIR(Room Impulse Response) 생성 및 에너지 정규화**
  - 독립된 좌/우 가우시안 백색 잡음($w_L, w_R$)에 감쇄 포락선을 곱해 스테레오 반사음 합성:
    $$\text{RIR}_L[t] = w_L[t] \cdot e(t), \qquad \text{RIR}_R[t] = w_R[t] \cdot e(t)$$
    $$\widetilde{\text{RIR}}_c = \frac{\text{RIR}_c}{\|\text{RIR}_c\|_2 + \epsilon} \quad (c \in \{L, R\}) \quad \to \quad \text{잔향 부가로 인한 원치 않는 볼륨 증가 방지}$$

---

* **3단계: 주파수 영역 고속 FFT 컨볼루션 (Fast Convolution)**
  - 시간 영역 직접 연산 대신 주파수 영역 곱셈을 수행하여 실시간 처리 속도 확보:
    $$y_{\text{wet}, c} = \text{iFFT}\Big( \text{FFT}(x_{\text{pad}}) \cdot \text{FFT}(\widetilde{\text{RIR}}_c) \Big)$$
* **4단계: Dry / Wet 선형 믹싱 (최종 스테레오 출력)**
  - 원음(Dry)과 잔향(Wet)을 지정된 비율($\text{wet} \in [0, 1]$)로 합산:
    $$y_{\text{rev}, c}[n] = (1 - \text{wet}) \cdot x_{\text{pad}}[n] + \text{wet} \cdot y_{\text{wet}, c}[n]$$

---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 & 플러그인 설계</span></div>
  <div class="toc-item active"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 11. 톤(Tone) 최적화: Mel-Spectrogram 스펙트럼 손실 & 정규화

EQ 파라미터($\theta_{\text{eq}} \in \mathbb{R}^{17}$)를 조율하여 내 보컬의 음색을 레퍼런스 보컬과 일치시키는 최적화 기준:

### 1) Mel-Spectrogram 스펙트럼 톤 손실 ($\mathcal{L}_{\text{mel}}$)
멜 스케일로 유효 대역($\Omega = [200\text{ Hz}, 10\text{ kHz}]$) 내의 로그 스펙트럼 오차 산출:
$$\mathcal{L}_{\text{mel}} = \frac{1}{|\Omega|} \sum_{m \in \Omega} \Big| \log_{10}(\Psi P_{\text{out}} + \epsilon)_m - \log_{10}(\Psi P_{\text{ref}} + \epsilon)_m \Big|$$
* $\Psi \in \mathbb{R}^{M \times F}$: 선형 주파수를 80개 멜 밴드로 매핑하는 Mel 필터뱅크 행렬.
* $P_{\text{out}}, P_{\text{ref}}$: 출력 보컬 및 레퍼런스 보컬의 파워 스펙트럼.

### 2) 음향학적 정규화 손실 (과도 왜곡 및 빗살 필터 방지)
수학적 손실만 추종하다가 특정 주파수가 뾰족하게 치솟는 "기계적 위화감"을 억제하기 위한 2중 물리 제약:
$$\mathcal{L}_{\text{reg}} = \lambda_2 \|\theta_{\text{eq}}\|_2^2 + \lambda_s \|\Delta \theta_{\text{eq}}\|_2^2 \qquad (\lambda_2 = 0.1, \; \lambda_s = 0.1)$$
* **과도 부스트 억제 ($\ell_2$)**: $\|\theta_{\text{eq}}\|_2^2 = \sum_{j=0}^{16} \theta_j^2 \quad \to \quad$ 불필요하게 극단적인 게인 증폭 방지.
* **인접 밴드 평활도 ($\ell_s$)**: $\|\Delta \theta_{\text{eq}}\|_2^2 = \sum_{j=0}^{15} (\theta_{j+1} - \theta_j)^2 \quad \to \quad$ 이웃한 필터 간의 급격한 꺾임을 막아 매끄러운 자연스러운 톤 유지.

---

## 12. 다이내믹스(Dynamics) 최적화: Crest Factor 손실

컴프레서 임계값(Threshold $T$)을 조율하여 내 보컬의 다이내믹 레인지를 레퍼런스와 일치시키는 최적화 기준:

### 1) 크레스트 팩터 (Crest Factor, CF)의 정의
신호의 **최대 피크(Peak)**와 **평균 에너지(RMS)** 사이의 음량 차이(비율):
$$\text{CF}(y) = \text{True Peak}_{\text{dB}}(y) - \text{RMS}_{\text{dB}}(y)$$
* **음향학적 의미**: 소리가 얼마나 순간적으로 튀는지(Spike), 아니면 고르게 압축되어 있는지(Dense)를 나타내는 다이내믹 레인지 척도.
  - **CF가 큰 소리**: 숨소리부터 샤우팅까지 편차가 큰 날것의 원본 보컬.
  - **CF가 작은 소리**: 컴프레서로 다이내믹스가 정돈되어 반주 위에 안정적으로 얹히는 상용 보컬.

### 2) 다이내믹스 최적화 손실 함수 ($\mathcal{L}_{\text{dyn}}$)
컴프레서가 신호를 누를수록 CF 수치는 왜곡 없이 **단조 감소(Monotonicity)**하여 역전파 최적화에 완벽하게 부합:
$$\mathcal{L}_{\text{dyn}} = |\text{CF}(y_{\text{comp}}) - \text{CF}(r)|$$
* **레벨 무관 특성 (Level-invariance)**: 컴프레서 내부의 오토 메이크업 게인은 상수 dB 오프셋이므로 Crest Factor($\text{Peak} - \text{RMS}$)에는 영향을 주지 않음 $\to$ 순수한 다이내믹스 파형의 조밀도만 독립적으로 정확히 최적화.
* **효과**: 내 보컬의 다이내믹스($\text{CF}$)를 레퍼런스 보컬과 정밀하게 일치시켜, **엔지니어가 수동으로 맞추듯 자연스럽고 일정한 음압**을 자동 실현.

---

## 13. 통합 목적 함수 & 상호 간섭 차단 (Selective Gradient)

음색과 다이내믹스를 동시에 조율할 때 두 플러그인이 서로 충돌하지 않도록 보장하는 핵심 공학 설계:

### 1) 전체 통합 목적 함수 (Total Objective)
$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{mel}} + \lambda_2 \|\theta_{\text{eq}}\|_2^2 + \lambda_s \|\Delta \theta_{\text{eq}}\|_2^2}_{\text{EQ 최적화 (톤 밸런스)}} + \underbrace{\mathcal{L}_{\text{dyn}}}_{\text{Compressor 최적화 (다이내믹스)}}$$

### 2) 핵심 제어 전략: 경사 분리 (`selective gradient`)
* **상호 간섭 문제**: 컴프레서가 신호를 누르면 스펙트럼 크기도 변하므로, 컴프레서 동작이 EQ 학습을 흔들고 EQ 부스트가 컴프레서 동작점을 틀어버리는 **파라미터 진동(Oscillation) 문제** 발생.
* **해결책**:
  $$\frac{\partial \mathcal{L}_{\text{mel}}}{\partial \theta_{\text{comp}}} \leftarrow 0, \qquad \frac{\partial \mathcal{L}_{\text{dyn}}}{\partial \theta_{\text{eq}}} \leftarrow 0$$
  - **톤 손실($\mathcal{L}_{\text{mel}}$)은 오직 EQ 파라미터($\theta_{\text{eq}}$)로만 역전파**.
  - **다이내믹스 손실($\mathcal{L}_{\text{dyn}}$)은 오직 컴프레서 Threshold로만 역전파**.
* **효과**: 두 모듈이 서로의 영역을 침범하지 않고 **독립적이며 완벽한 단조 수렴(Monotonic Convergence)**을 달성.

---

## 14. End-to-End 경사하강법 워크플로우

```
[입력 보컬] ──> [ 미분 가능 EQ ] ──> [ 미분 가능 Comp ] ──> [ Reverb ] ──> [ 최종 출력 ]
                      ▲                     ▲                                    │
                      │                     │                                    ▼
                      └───────────── 역전파 (Backprop) ────────────────────── Loss 계산
```

* **손실 측정 지점 (`post_reverb`)**:
  - EQ, 컴프레서뿐만 아니라 공간 잔향(Reverb)까지 모두 반영된 **최종 출력 음원에서 Loss를 계산**하여, 실제 청취 환경 기준의 완벽한 밸런스를 유도.
* **최적화 1회 반복 사이클 (Iteration Cycle)**:
  - 초기 상태: $\theta_{\text{eq}} = 0$ (Flat EQ), Threshold = $-30\text{ dB}$, Ratio = $10:1$
  - **1) 순전파 (Forward)**: EQ $\to$ Comp $\to$ Reverb를 거쳐 최종 스테레오 출력 생성
  - **2) 손실 계산 (Loss)**: 톤 오차($\mathcal{L}_{\text{mel}}$) + 다이내믹스 오차($\mathcal{L}_{\text{dyn}}$) + 정규화 손실 합산
  - **3) 역전파 (Backward)**: 그래디언트 계산 (Selective Gradient로 상호 간섭 차단)
  - **4) 파라미터 갱신 (Update)**: Adam Optimizer로 파라미터 업데이트 ($\text{lr}_{\text{eq}}=\text{lr}_{\text{comp}}=0.05$, 전 모듈 동일)
* **안정적인 수렴 (50회 반복)**:
  - 수 시간~수 일이 소요되는 거대 딥러닝 모델 학습과 달리, **단 50회의 가벼운 반복 최적화(50 Iterations)**로 손실이 최적점으로 빠르게 수렴.
  - 곡 전체 처리 기준 **약 1~3분 내외 소요** (일반 CPU 환경 기준).

---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 & 플러그인 설계</span></div>
  <div class="toc-item"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item active"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 15. 풀믹스 합산: 인간 청각 인지 라우드니스(LUFS) 매칭

보컬과 반주를 합칠 때 곡 전체 평균을 쓰면, 반주의 조용한 인트로/아웃트로 때문에 후렴구에서 보컬 밸런스가 완전히 무너짐:

### 1) 인간 청각 인지 라우드니스: LUFS (ITU-R BS.1770-4)
단순 피크(Peak)나 RMS 대신 인간의 귀가 민감하게 반응하는 고역/저역을 보정한 국제 표준 라우드니스:
* **K-Weighting 필터**: 머리 형상 효과($+4\text{ dB}$) 및 저역 둔감도 보정.
* **2단 게이팅**: 무음($-70\text{ LUFS}$) 및 조용한 구간($-10\text{ LU}$)을 통계에서 제외하여 핵심 소리 구간만 측정.
$$\text{LUFS} = -0.691 + 10\log_{10}\left( \frac{\sum_j g_j z_j}{\sum_j g_j} \right)$$

### 2) 3초 슬라이딩 윈도우 최대 LUFS 정렬 & 헤드룸 보호
곡에서 가장 에너지가 폭발하는 하이라이트(후렴구) 구간의 음압을 정확히 타깃:
* **가장 큰 3초 구간 탐색**: $\text{LUFS}_{\text{loudest}}(x) = \max_{k} \text{LUFS}\big( x[k f_s : k f_s + 3 f_s] \big)$
* **보컬 목표 레벨 정렬**: $\Delta G_{\text{dB}} = \text{LUFS}_{\text{loudest}}(\text{Inst}) - \text{LUFS}_{\text{loudest}}(\text{Vocal}) \quad \to \quad v_{\text{scaled}} = v \cdot 10^{\Delta G_{\text{dB}} / 20}$
* **클리핑 방지**: 헤드룸 $-1.0\text{ dBFS}$ 초과분만 $\tanh$로 부드럽게 완충 ($y_{\text{mix}} = \text{SoftLimit}(v_{\text{scaled}} + \text{Inst})$)

---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 & 플러그인 설계</span></div>
  <div class="toc-item"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item active"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 16. 마스터링(Mastering)이란 무엇인가?

### 1) 마스터링의 정의와 목적
* **정의**: 개별 트랙들을 합친 믹스다운(Mixdown) 트랙을 모든 상용 스트리밍 플랫폼(Spotify, Apple Music, YouTube) 및 다양한 스피커 환경에서 **최고의 음질과 균일한 음압**으로 재생되도록 가공하는 최종 상용화 단계.
* **핵심 작업 3요소**:
  1. **전체적 톤 밸런스 정돈 (Mastering EQ)**: 특정 악기가 튀지 않도록 전체 주파수 균형 완성.
  2. **상용 음압(Loudness) 극대화**: 상용 음원들과 경쟁 가능한 음압 확보.
  3. **디지털 피크 제어 (Brickwall Peak Limiting)**: $0\text{ dBFS}$를 넘는 클리핑 차단.

---

## 17. 마스터링 1단계: Mastering EQ (`master_track`)

풀믹스의 멜 스펙트럼 포락선을 레퍼런스 곡에 매칭 (`modules.py:828-863` 구현 기반):

### 1) 믹싱 EQ vs 마스터링 EQ 파라미터 비교

| 파라미터 | 보컬 믹싱 EQ (`DifferentiableEQ`) | 마스터링 EQ (`master_track`) | 코드 위치 / 설정 근거 |
|---|---|---|---|
| **처리 대상** | 단일 보컬 트랙 | 풀믹스 트랙 | `modules.py:812` |
| **주파수 대역** | **$200\text{ Hz} \sim 10\text{ kHz}$** | **$20\text{ Hz} \sim 10\text{ kHz}$** | `modules.py:117-120, 835` |
| **밴드 수 (`num_bands`)** | 17-Band | 15-Band | `modules.py:805` |
| **게인 스케일 계수 (`gain_scale_db`)** | $15.0\text{ dB}$ (기울기 스케일 $d/d\theta$) | $8.0\text{ dB}$ (기울기 스케일 $d/d\theta$) | `modules.py:163, 806` |
| **클램프 (`hard_clamp`)** | `False` (선형 개방) | `False` (선형 개방) | `modules.py:833` 주석 |
| **L2 페널티 (`eq_l2`)** | $0.1$ | **$1.0$** | `modules.py:807` 주석 |
| **스무딩 페널티 (`eq_smooth`)** | $0.1$ | **$1.0$** | `modules.py:807` 주석 |
| **학습률 (`lr_eq`)** | `0.05` (실제 변위 $\approx 0.75\text{ dB/step}$) | **`0.05`** (실제 변위 $\approx 0.40\text{ dB/step}$) | `pipeline.py:831`, `modules.py:807, 841` |
| **반복 횟수 (`eq_iters`)** | 50 (E2E 전체 파이프라인) | **300** (스펙트럼 매칭 루프) | `modules.py:806` |

---

### 2) 왜 마스터링 EQ는 $20\text{ Hz} \sim 10\text{ kHz}$ 대역을 타깃하는가?

* **저역 ($20\text{ Hz} \sim 200\text{ Hz}$)을 온전히 살리는 이유 (킥/베이스 보존)**:
  - 믹싱에서는 보컬 분리 AI의 베이스 마스킹 회피를 위해 저역을 배제했으나, 마스터링은 **풀믹스 원곡 레퍼런스**를 사용하므로 킥 드럼과 서브베이스가 온전히 보존되어 있어 저역 톤 밸런스를 $20\text{ Hz}$부터 정밀하게 매칭함.
* **고역 상한선을 $10\text{ kHz}$로 제한하는 이유 (코덱 절벽 방지)**:
  - 레퍼런스가 MP3/AAC 등 손실 압축 음원일 경우 $15\sim 16\text{ kHz}$ 이상에서 주파수가 잘려나가는 **코덱 절벽(Codec Cliff)** 현상이 존재함.
  - 평가 대역을 $10\text{ kHz}$ 이하(`mel_mask`)로 한정하여, 레퍼런스의 코덱 압축 결함을 따라 믹스의 에어감(Air Band)을 부당하게 깎아버리는 음향 왜곡을 원천 차단함 (`modules.py:833`).

---

## 18. 마스터링 2단계: 보컬 믹싱 컴프 vs 마스터링 리미터 비교

풀믹스 다이내믹 제어 및 상용 음압(LUFS) 매칭 (`modules.py:882-960` 구현 기반):

### 1) 믹싱 컴프 vs 마스터링 리미터 파라미터 비교

| 파라미터 | 보컬 믹싱 컴프 (`DifferentiableCompressor`) | 마스터링 리미터 (`DifferentiableLimiter`) | 코드 위치 / 설정 근거 |
|---|---|---|---|
| **처리 대상** | 단일 보컬 트랙 (모노/스테레오) | 풀믹스 2트랙 (`mix = vocal + inst`, 스테레오 링크) | `pipeline.py:624`, `modules.py:715` |
| **핵심 목적** | 보컬 다이내믹 레인지 정돈 & Crest Factor 매칭 | 상용 음압(LUFS) 확보 & $0\text{ dBFS}$ 클리핑 방지 | `losses.py:328`, `modules.py:802` |
| **제어 파라미터** | **Threshold ($T_{\text{comp}}$) 하강** | **입력 드라이브 게인 ($G_{\text{in}}$) 상승** | `pipeline.py:702`, `modules.py:906` |
| **천장/기준점** | 가변 Threshold ($T \in [-30, 0]\text{ dB}$) | **Ceiling 고정 ($-0.3\text{ dBFS}$)** | `modules.py:794, 887` |
| **압축 비율 (`ratio`)** | **$10.0:1$** (보컬 다이내믹 압축) | **$50.0:1$** (브릭월 피크 리미팅) | `pipeline.py:149`, `modules.py:796` |
| **어택 타임 (`attack_ms`)** | $0.5\text{ ms}$ (보컬 트랜지언트 엔벨로프 추종) | **$0.1\text{ ms}$** (순간적 인터샘플 피크 즉각 차단) | `pipeline.py:602`, `modules.py:796` |
| **릴리즈 타임 (`release_ms`)** | $120.0\text{ ms}$ (자연스러운 보컬 서스테인 복원) | **$80.0\text{ ms}$** (믹스 펌핑 방지 및 빠른 복원) | `pipeline.py:603`, `modules.py:796` |

---

| 파라미터 | 보컬 믹싱 컴프 (`DifferentiableCompressor`) | 마스터링 리미터 (`DifferentiableLimiter`) | 코드 위치 / 설정 근거 |
|---|---|---|---|
| **소프트 니 (`knee_db`)** | $2.0\text{ dB}$ | **$1.0\text{ dB}$** (천장 경계 정밀 완충) | `modules.py:276, 797` |
| **최적화 손실 함수** | $\mathcal{L}_{\text{dyn}} = \|\text{CF}_{\text{src}} - \text{CF}_{\text{ref}}\| + \lambda_{\text{th}}\text{ReLU}(T - T_{\text{max}})^2$ | $\mathcal{L} = \text{ReLU}(G_{\text{target}} - G_{\text{in}})^2 + 25.0 \cdot \text{ReLU}(\max(\text{GR}) - 3.0)^2$ | `losses.py:328`, `modules.py:928-932` |
| **게인 보상 / 안전 장치** | 활성 구간 평균 GR 계산 후 deterministic offset | **3.0 dB GR 상한선 페널티 + 3.5 dB 하드 캡 가드** | `pipeline.py:734`, `modules.py:943-953` |
| **학습률 / 스텝** | `lr_comp = 0.05`, 50 스텝 E2E (모듈별 분리 그래디언트) | **`lr = 0.05`, 800 스텝** 단일 학습률 Adam | `pipeline.py:832`, `modules.py:749-750, 921` |

---

### 2) 왜 Threshold가 아닌 입력 드라이브(볼륨)를 올리는가?

* **천장(Ceiling) $-0.3\text{ dBFS}$ 고정의 원칙 (`modules.py:794`)**:
  - DAC(디지털-아날로그 변환기) 재생 시 인터샘플 피크(Inter-sample Peak) 클리핑을 막기 위해 Ceiling은 $-0.3\text{ dBFS}$에 단단히 고정되어야 함.
* **Threshold 하강 vs 입력 드라이브 상승의 메커니즘 차이**:
  - **보컬 믹싱**: 트랙 자체의 동적 편차를 줄이기 위해 Threshold를 낮추고, 압축으로 깎인 레벨은 활성 구간 평균 게인 리덕션($\overline{\text{GR}}_{\text{active}}$)만큼 들어 올려 보상함.
  - **마스터링**: 천장($-0.3\text{ dBFS}$)을 고정한 채 **`drive_gain`($G_{\text{in}}$)으로 신호를 밀어 올려 상용 음압(LUFS)을 채우고**, 천장을 넘어서는 극단적 피크만 $50:1$ 비율로 투명하게 리미팅함 (`modules.py:882-960`).

---

## 목차 (Table of Contents)

<div class="toc-container">
  <div class="toc-item"><span class="num">1</span><span class="title">믹스의 정의 &amp; 시스템 개요</span></div>
  <div class="toc-item"><span class="num">2</span><span class="title">핵심 도구 3종의 수학적 모델링 & 플러그인 설계</span></div>
  <div class="toc-item"><span class="num">3</span><span class="title">최적화 메커니즘 (Optimization Mechanism)</span></div>
  <div class="toc-item"><span class="num">4</span><span class="title">풀믹스 합산 (Summing with Instrumental)</span></div>
  <div class="toc-item"><span class="num">5</span><span class="title">마스터링 체인 (Mastering Chain)</span></div>
  <div class="toc-item active"><span class="num">6</span><span class="title">결론 (Conclusion)</span></div>
</div>

---

## 19. 결론 (Conclusion)

1. **패러다임 전환**:
   - 정답 데이터셋이 없는 실제 음악 제작 환경에서, **단 하나의 레퍼런스 곡만을 이용해 최적의 믹싱·마스터링 값을 도출하는 비지도 최적화 파이프라인**을 완성함.
2. **수학적 엄밀성과 물리 음향학의 결합**:
   - 전 구간 미분 가능한 형태로 설계하여 그래디언트 소실 없이 End-to-End 경사하강법을 구현함.
   - 음원 분리 ai의 한계점인 마스킹(200Hz) 및 코덱 절벽(10kHz)을 공학적으로 격리하여 ai의 한계 개선.


---

# Q & A

감사합니다.
질문이 있으시면 자유롭게 말씀해 주시기 바랍니다.
