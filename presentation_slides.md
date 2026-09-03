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
    padding: 34px 48px;
    background-color: #ffffff;
    color: #1f2937;
    font-size: 25px;
  }
  h1 { color: #1e3a8a; font-size: 1.85em; margin-bottom: 0.3em; font-weight: 700; }
  h2 { color: #1e40af; border-bottom: 2px solid #e5e7eb; padding-bottom: 6px;
       font-size: 1.22em; margin-top: 0; font-weight: 600; }
  h3 { color: #2563eb; font-size: 1.0em; margin-bottom: 6px; font-weight: 600; }
  footer { color: #6b7280; font-size: 0.55em; }
  code { background-color: #f3f4f6; color: #1e40af; padding: 2px 6px; border-radius: 4px;
         font-family: 'Fira Code', 'Consolas', monospace; border: 1px solid #e5e7eb; }
  blockquote { background: #fff7ed; border-left: 4px solid #ea580c; padding: 10px 16px;
               margin: 10px 0; color: #7c2d12; border-radius: 0 6px 6px 0; font-size: 0.92em; }
  ul, ol { margin-top: 6px; margin-bottom: 6px; font-size: 0.9em; line-height: 1.5; }
  li { margin-bottom: 3px; }
  strong { color: #1d4ed8; font-weight: 600; }
  table { font-size: 0.75em; }
---

# 발표

DDSP Vocal Auto-Mix 

이현중 · 2026.09.02


---

## 1. 범위와 기호

**범위**: 입력 로드 → EQ → 컴프 → 손실 → 지표. 리버브 **모듈 내부**(`MeasuredIRReverb` 의 IR 추정, `L_decay`)는 이 문서 밖이다. 측정 IR 리버브는 학습 파라미터가 없으므로 그 경우 `L_decay` 는 **계산 자체를 하지 않는다**(합산 구조가 아니므로 가중치로 끄는 개념도 없다).

> **단, 리버브는 손실 경로 안에 있다** (`LOSS_MEASURE_POINT = "post_reverb"`). $L_{\text{tone}}$·$L_{\text{dyn}}$ 은 리버브까지 통과한 신호에서 재고, EQ·컴프의 그래디언트가 리버브 컨볼루션을 **관통해서** 흐른다. 리버브 자체에 학습할 스칼라가 없다는 것과, 리버브가 손실이 보는 신호를 바꾼다는 것은 별개다.

---

| 기호 | 의미 |
|---|---|
| $x[n]$, $r[n]$ | 내 보컬 / 레퍼런스 보컬 (mono, $f_s=44100$) |
| $S(f,t)$, $P=\lvert S\rvert^2$ | STFT / 파워 스펙트로그램 |
| $F = 1025$ | rfft bin 수 ($N_{\text{fft}}/2+1$) |
| $\Psi\in\mathbb{R}^{80\times F}$ | mel 필터뱅크 (slaney) |
| $\Omega$ | 평가 mel 밴드 집합 $\{m: 200 \le f_c(m) < 10000\}$ (61개) |
| $\theta\in\mathbb{R}^J$ | EQ 밴드 파라미터 |
| $\theta_T$ | 컴프 threshold 파라미터 (ratio 는 3:1 상수) |
| $\alpha$ | 매칭 강도 (`match_amount`) |
| $\rho$ | 컴프 적용량 (`comp_amount`) |

---

## 2. 상수표 (코드에서 그대로)

| 구분 | 기호 | 값 | 위치 |
|---|---|---|---|
| STFT | $N_{\text{fft}},H$ | 2048, 512 (Hann, center) | `pipeline.py` |
| EQ | $J$ | **17** (함수·API·UI 공통 기본) | `pipeline.VOCAL_EQ_BANDS` |
| EQ | $G_{\max}$ | 15 dB | `pipeline.match_e2e(max_gain_db=)` |
| EQ | 밴드 범위 | **200 Hz – 10 kHz** (로그 등간격) | `VOCAL_EQ_MIN/MAX_FREQ` |
| EQ | $\sigma$ | $0.6\times$ 밴드 간격 (log2) = 0.2116 oct | `modules.py` |
| EQ 마스크 | $M_{\text{eq}}$ | $[200, 10000)$ Hz 밖은 0 dB 바이패스 | `pipeline.py` |
| 컴프 | $K$ | 4 dB (soft knee) | `DifferentiableCompressor(knee_db=)` |
| 컴프 | $H_d, W_d$ | 128, 256 samples | `DifferentiableCompressor(detector_hop=)` |
| 컴프 | $\tau_a,\tau_r$ | 0.5 ms, 120 ms | `pipeline.match_e2e(attack_ms=, release_ms=)` |
| 컴프 | $R$ | **설정 가능 상수 (기본 3.0 : 1), 학습 안 함** | `pipeline.COMP_RATIO` |
| 컴프 초기값 | $T_0$ | $-30$ dB | `modules.py` |

---

| 구분 | 기호 | 값 | 위치 |
|---|---|---|---|
| 정규화 | $\lambda_2,\lambda_s$ | 0.1, 0.1 | `main.py` |
| 정규화 | $w_t$ | 0.0 (threshold 심도) | `main.py` |
| 학습 | lr | EQ 0.05 / Comp 0.02, $n=50$ | `pipeline.match_e2e(lr_eq=, lr_comp=, n_steps=)` |
| 학습 | 그래디언트 경로 | `"selective"` | `pipeline.LOSS_GRAD_MODE` (§11) |
| 학습 | 손실 측정 지점 | `"post_reverb"` (리버브 뒤) | `pipeline.LOSS_MEASURE_POINT` |

---

## 3. 전처리와 분석

**피크 정규화** (두 신호 모두):

$$x \leftarrow 0.501187\cdot\frac{x}{\max_n\lvert x[n]\rvert}\quad(-6\ \text{dBFS}),\qquad r \leftarrow 0.501187\cdot\frac{r}{\max_n\lvert r[n]\rvert}$$

**STFT / iSTFT** (Hann 창 $w$, `center=True`, reflect 패딩):

$$S(f,t)=\sum_{n} x[n+tH]\,w[n]\,e^{-i2\pi fn/N_{\text{fft}}},\qquad y=\text{iSTFT}(S)$$

창과 홉이 COLA 조건을 만족하므로 $\text{iSTFT}(\text{STFT}(x))=x$ (경계 제외).

**학습 구간**은 손실마다 다르다.

* $L_{\text{tone}}$ (EQ): **곡 전체** ($x_{\text{train}}=x$). 짧은 구간을 쓰면 EQ 가 그 구간에 과적합되어 곡 전체 톤이 악화된다(실측 2.419 vs 무처리 2.369).
* $L_{\text{dyn}}$ (컴프): **하이라이트 15초 구간**. 실제 믹싱에서 가장 큰 대목을 기준으로 컴프를 잡는 것과 같은 정의다. 지표도 같은 구간에서 잰다 → §22-B.


---

## 4. EQ — 밴드 배치

밴드 중심 주파수는 $[200,\,10000]$ Hz 로그 등간격, $J=17$:

$$\boxed{\ f_j = 200\cdot 50^{\frac{j}{J-1}},\qquad j=0,\dots,J-1\ }$$

$c_j=\log_2 f_j$ 로 두면 간격은 옥타브 단위로 **균일**(constant-Q):

$$\Delta = \frac{\log_2 50}{J-1}\ \overset{J=17}{=}\ 0.3527\ \text{oct},\qquad \sigma=0.6\Delta=0.2116\ \text{oct}$$

필터뱅크 행렬 $\Phi\in\mathbb{R}^{F\times J}$, 학습 파라미터 $\theta\in\mathbb{R}^{17}$:

$$\Phi(f,j)=\exp\!\left(-\frac{(\log_2 f - c_j)^2}{2\sigma^2}\right)$$

---

**배치 (Hz)**
$$200,\;255,\;326,\;416,\;532,\;679,\;867,\;1107,\;1414,\;1806,\;2306,\;2945,\;3761,\;4802,\;6132,\;7831,\;10000$$

FWHM $=2\sqrt{2\ln2}\,\sigma = 0.498$ oct.

**왜 이 범위인가** — 밴드 범위를 손실의 평가 대역 $\Omega$ 와 **정확히 일치**시킨다. 평가되지 않는 곳에 파라미터를 두면 그래디언트 없이 표류하고, 파라미터가 없는 곳을 평가하면 고칠 수 없는 오차만 쌓인다.

* **하한 200 Hz**: 보컬 분리(source separation) 과정에서 $100$–$150$ Hz 의 보컬이 악기에 **마스킹**되어, 분리 모델이 그 대역 에너지를 복원하지 못한다. 노이즈가 섞인 정도가 아니라 **정보 자체가 소실**된 구간이라 매칭 타겟으로 쓸 수 없다.
* **상한 10 kHz**: 손실 압축(mp3 등) 레퍼런스는 이 위 정보가 대부분 소실돼 있다. 실측 (`Golden Acapella.mp3`, 정규화 mel dB):

| 중심 | 13.9 k | 16.3 k | 17.1 k | 18.0 k | 19.9 k |
|---|---|---|---|---|---|
| raw − ref | $-14.9$ | $-4.9$ | $\mathbf{+43.3}$ | $\mathbf{+50.5}$ | $\mathbf{+45.6}$ |

17 kHz 위의 $+43$~$+50$ dB 는 **코덱 절벽**이지 톤 차이가 아니다. 손실은 그것을 구분하지 못하고 EQ 에 거대한 컷을 요구했다.

---

**배치 변천**

| | 범위 | $J$ | $\Delta$ | $\sigma$ | 비고 |
|---|---|---|---|---|---|
| 최초 | 20–20 k | 30 | 0.3436 oct | 0.2062 |  |
| | 100–20 k | 23 | 0.3474 oct | 0.2085 | 고역 코덱 절벽 노출 |
| **현재** | **200–10 k** | **17** | **0.3527 oct** | **0.2116** | 분리 소실 저역 제외 |


---

## 5. EQ — 곡선과 적용

**밴드 게인 → 곡선** (E2E 는 `hard_clamp=False`):

$$\gamma_j = G_{\max}\,\theta_j\ \ (\text{상한 없음}),\qquad G_{\text{dB}}(f)=\sum_{j=0}^{J-1}\Phi(f,j)\,\gamma_j$$

**마스크는 하나뿐이다** — 대역 바이패스:

$$M_{\text{eq}}(f)=\mathbb{1}\big[200 \le f < 10000\ \text{Hz}\big]$$

**최종 적용** (복소 STFT 에 실수 게인 → 위상 보존):

$$\boxed{\ \mathcal{G}(f)=10^{\,\alpha\,G_{\text{dB}}(f)M_{\text{eq}}(f)/20}\ },\qquad S'(f,t)=\mathcal{G}(f)S(f,t)$$


---

## 6. 컴프 — 피크 검출기

시간 영역, 홉 $H_d=128$ ($2.9$ ms), 창 $W_d=256$:

$$e_k=\max_{i\in[kH_d,\;kH_d+W_d)}\big\lvert u[i]\big\rvert,\qquad E_k=20\log_{10}(e_k+\epsilon)$$

여기서 $u$ 는 컴프의 입력이다. 학습 시 컴프는 **두 번** 호출되고 $u$ 가 서로 다르다 — 톤 경로는 $u=y_{\text{eq}}$ (그래프 연결, threshold 만 $\text{sg}$), 다이내믹 경로는 $u=\text{sg}[y_{\text{eq}}]$ 다. 렌더는 $u=y_{\text{eq}}$ (§10, §11).

검출기 레이트 $f_{\text{det}}=f_s/H_d=344.53$ Hz.


패딩: $\text{pad}=(W_d-(N\bmod H_d))\bmod H_d + W_d$ 만큼 뒤에 0 을 붙여 마지막 프레임까지 창이 채워지게 한다.

---

## 7. 컴프 — 게인 계산 (soft knee)

$d_k = E_k - T$ 에 대해

$$
o_k=\begin{cases}
0,& d_k<-\tfrac{K}{2}\\[6pt]
\dfrac{\big(d_k+\tfrac{K}{2}\big)^2}{2K},& \lvert d_k\rvert\le \tfrac{K}{2}\\[10pt]
d_k,& d_k>\tfrac{K}{2}
\end{cases}
\qquad K=4\ \text{dB}
$$

연속성 확인: $d=-K/2$ 에서 $o=0$, $d=+K/2$ 에서 $o=K/2=d$. 도함수도 $0\to 1$ 로 연속.

**목표 게인 리덕션**:

$$G^{\text{tgt}}_k=-\,o_k\Big(1-\frac1R\Big)\ \le 0$$

**파라미터 사상** — **학습 변수는 threshold 하나뿐이고 ratio 는 상수다**:

$$T=-60\,\sigma(\theta_T)\in[-60,0]\ \text{dB},\qquad R \equiv \text{const}\$$


---

## 8. 컴프 — 밸리스틱 (전방)

$$G_k=a_kG_{k-1}+(1-a_k)G^{\text{tgt}}_k,\qquad G_{-1}=0$$

$$a_k=\begin{cases}a_{\text{att}},&G^{\text{tgt}}_k<G_{k-1}\ (\text{더 누르는 중})\\ a_{\text{rel}},&\text{그 외 (회복 중)}\end{cases}$$

$$a_\bullet=\exp\!\left(-\frac{1}{\tau_\bullet f_{\text{det}}}\right)
\quad\Rightarrow\quad a_{\text{att}}=e^{-1/0.172}=0.0031,\quad a_{\text{rel}}=e^{-1/41.3}=0.9761$$

($\tau_a=0.5$ ms $\Rightarrow \tau_a f_{\text{det}}=0.172$ 프레임, $\tau_r=120$ ms $\Rightarrow 41.3$ 프레임)

이 재귀가 없으면 $G_k=G^{\text{tgt}}_k$ 로 시정수가 존재하지 않아 **컴프가 아니라 게인 오토메이션**이 된다.

시정수는 **상수 버퍼**다. 학습 대상으로 만들면 $(T,R)$ 과 강하게 결합해 손실 지형만 나빠진다.

---

## 9. 컴프 — 밸리스틱 (역전파 유도)

분기 $\{a_k\}$ 를 고정하면 재귀는 선형이다. $G_k$ 가 $G_{k+1}$ 에만 직접 영향을 주므로, 손실 $L$ 에 대해

$$
s_k \;\equiv\; \frac{\partial L}{\partial G_k}\Big|_{\text{전체}}
\;=\; \frac{\partial L}{\partial G_k}\Big|_{\text{직접}} + \frac{\partial L}{\partial G_{k+1}}\cdot\frac{\partial G_{k+1}}{\partial G_k}
\;=\; g_k + a_{k+1}s_{k+1}
$$

역방향으로 한 번 훑으면 모든 $s_k$ 가 나오고,

$$\boxed{\ \frac{\partial L}{\partial G^{\text{tgt}}_k}=(1-a_k)\,s_k\ }$$

**비용**: $O(M)$ 시간·메모리. 파이썬 루프를 autograd 그래프로 태우면 깊이 $M$ 의 그래프가 생겨 실측 **34분**(15 s 구간·200 스텝)이 걸렸다. 위 수반 재귀로 대체해 $O(M)$.

**분기 고정의 정당성**: $a_k$ 는 $\mathbb{1}[G^{\text{tgt}}_k<G_{k-1}]$ 로 결정되는 이산 선택이고, 이 지표함수는 경계집합(측도 0)을 제외하면 미분이 $0$ 이다. 따라서 backward 에서 분기를 상수로 두는 것은 **근사가 아니라 정확한 미분**이다.

---

## 10. 컴프 — 메이크업과 적용

활성 집합 $\mathcal{A}=\{k:E_k>\max_j E_j-45\}$ 에 대해

$$G^{\text{net}}_k=G_k-\frac{1}{\lvert\mathcal{A}\rvert}\sum_{k\in\mathcal{A}}G_k$$

**샘플 레이트 복원** (선형 보간) 후 곱. 학습 시 $u$ 와 $T$ 중 어느 쪽이 $\text{sg}$ 를 받는지는 경로마다 다르다:

$$
y_{\text{comp}}[n]=u[n]\cdot 10^{\,\text{interp}(G^{\text{net}})[n]/20},
\qquad
(u,\,T)=\begin{cases}
(y_{\text{eq}},\ \text{sg}[\theta_T]) & \text{(학습 · 톤 경로)}\\
(\text{sg}[y_{\text{eq}}],\ \theta_T) & \text{(학습 · 다이내믹 경로)}\\
(y_{\text{eq}},\ \theta_T) & \text{(렌더)}
\end{cases}
$$

검출기 $E_k$ 도 같은 $u$ 에서 뽑는다. 다이내믹 경로에서는 $u$ 가 끊겨 컴프 내부 전체가 EQ 그래프와 분리되지만, **톤 경로는 반대다** — $u$ 가 살아 있어 EQ 그래디언트가 컴프를 관통하고, 대신 $\text{sg}[\theta_T]$ 가 톤 손실이 컴프를 움직이는 것을 막는다(§11).

---

**적용량 블렌드** (병렬 컴프레션 형태):

$$y_{\text{dry}} = u + \rho\,(y_{\text{comp}}-u)$$

학습 중 $\rho=1$, 렌더 시 사용자 값 → **§18 감사 항목 A**.

리버브가 꺼져 있으면 체인 출력은 $\mathbf{y}=[y_{\text{dry}};\,y_{\text{dry}}]$ (스테레오 복제)이고, 손실은 채널 평균을 취하므로 $y_{\text{dry}}$ 와 동일하다.

---

## 11. 목적함수 — 같은 소리를 듣되, 경로는 가른다

손실은 하나의 스칼라로 합치지 않는다(가중치 이야기는 §16). 두 손실 모두 **체인 최종 출력**(컴프 → 리버브 통과 후)에서 재되, 그래디언트는 각자 자기 모듈로만 흐른다 — `LOSS_GRAD_MODE = "selective"`, `LOSS_MEASURE_POINT = "post_reverb"`.

$\text{Rev}(\cdot)$ 는 리버브 적용 후 채널 평균(`E2EChain.apply_reverb(·).mean(0)`)이다. 리버브가 꺼져 있으면 그대로 컴프 출력이 된다.

$$
y_{\text{eq}}=\text{EQ}(x;\theta),\qquad
\underbrace{\text{tone}_{\text{src}}=\text{Rev}\Big(\text{Comp}\big(y_{\text{eq}};\,\text{sg}[\theta_T]\big)\Big)}_{\theta\ \text{로만}},\qquad
\underbrace{\text{dyn}_{\text{src}}=\text{Rev}\Big(\text{Comp}\big(\text{sg}[y_{\text{eq}}];\,\theta_T\big)\Big)\Big|_{W}}_{\theta_T\ \text{로만}}
$$

$\text{Rev}$ 에는 $\text{sg}$ 가 걸리지 않는다 — **그래디언트가 리버브를 관통해야** EQ·컴프가 최종 출력 기준으로 정해진다. 리버브에 학습 파라미터가 없어도 컨볼루션은 선형이라 backward 가 성립한다. 자르기 $\big|_W$ 는 **리버브 뒤**다: 구간을 먼저 자르고 리버브를 걸면 구간 시작 앞 신호가 만든 잔향 유입이 빠져 앞부분이 실제 출력보다 드라이해진다(컴프 밸리스틱과 같은 이유).

$$
\mathcal{L}_{\text{tone}}(\theta)\;=\;L_{\text{tone}}\big(\text{tone}_{\text{src}}\big)\;+\;R_{\text{eq}}(\theta)
\qquad
\mathcal{L}_{\text{dyn}}(\theta_T)\;=\;L_{\text{dyn}}\big(\text{dyn}_{\text{src}}\big)\;+\;w_t\Big(\tfrac{T}{-60}\Big)^2
$$

---


**측정 구간**: 톤은 곡 전체(장기 평균 스펙트럼이 타겟, §3), 다이내믹은 하이라이트 15초 $W$(§22-B). 컴프는 두 경우 모두 **곡 전체**에 걸고 $L_{\text{dyn}}$ 만 출력에서 $W$ 를 잘라 쓴다 — 구간을 먼저 잘라 넣으면 밸리스틱 상태가 리셋되고 오토 메이크업이 구간 평균으로 계산돼 값이 어긋난다(실측 차이 $6.7\times10^{-2}$).

**학습 루프**
```
l_tone.backward()    # θ 에만 누적
l_dyn.backward()     # θ_T 에만 누적
optimizer.step()     # 단일 Adam, param group 으로 lr 분리
```

$$\text{Adam},\quad \eta_{\text{EQ}}=0.05\ (\theta),\qquad \eta_{\text{comp}}=0.02\ (\theta_T),\qquad n_{\text{steps}}=50$$



**초기값**: $\theta=0$ (평탄), $T_0=-30$ dB. $R$ 은 설정 상수(§7).

<!--
---

## 11-B. 그래디언트 경로 — 세 모드의 대비

**정지 그래디언트 연산자** $\text{sg}[\cdot]$ 는 항등 사상이되 미분이 0 이다:

$$\text{sg}[u]=u \quad(\text{forward}),\qquad \frac{\partial\,\text{sg}[u]}{\partial u}\equiv 0\quad(\text{backward})$$

`selective` 는 이것을 **두 곳에 서로 다르게** 건다.

$$
\frac{\partial \mathcal{L}_{\text{tone}}}{\partial\theta_T}
=\frac{\partial L_{\text{tone}}}{\partial\,\text{tone}_{\text{src}}}\cdot
\frac{\partial\,\text{Comp}(y_{\text{eq}};\,\text{sg}[\theta_T])}{\partial\,\text{sg}[\theta_T]}\cdot
\underbrace{\frac{\partial\,\text{sg}[\theta_T]}{\partial\theta_T}}_{=\,0}\;=\;0
$$

$$
\frac{\partial \mathcal{L}_{\text{dyn}}}{\partial\theta}
=\frac{\partial L_{\text{dyn}}}{\partial\,\text{dyn}_{\text{src}}}\cdot
\frac{\partial\,\text{Comp}(\text{sg}[y_{\text{eq}}];\theta_T)}{\partial\,\text{sg}[y_{\text{eq}}]}\cdot
\underbrace{\frac{\partial\,\text{sg}[y_{\text{eq}}]}{\partial y_{\text{eq}}}}_{=\,0}\cdot
\frac{\partial y_{\text{eq}}}{\partial\theta}\;=\;0
$$

**실측** (20초 랜덤 신호, backward 직후 누적된 그래디언트):

| 모드 | 톤 측정 지점 | $L_{\text{tone}}$ backward 후 $\theta_T$ | $L_{\text{dyn}}$ backward 후 $\theta_{\text{EQ}}$ 변화 |
|---|---|---|---|
| split | $y_{\text{eq}}$ (컴프 이전) | `None` | $+0.0000$ |
| **selective** | $y_{\text{full}}$ (컴프 이후) | **`None`** | $\mathbf{+0.0000}$ |
| unified | $y_{\text{full}}$ | $-5.28\times10^{-7}$ | $+0.4648$ |

`selective` 는 **split 과 같은 격리**를 유지하면서 **unified 와 같은 측정 지점**을 쓴다.

---

## 11-C. 최적화 동역학 — 세 모드의 야코비안

세 모드 모두 두 목적함수를 동시에 내려간다:

$$
\theta^{(k+1)}=\theta^{(k)}-\eta_1\widehat{\nabla}_\theta \mathcal{L}_{\text{tone}},\qquad
\theta_T^{(k+1)}=\theta_T^{(k)}-\eta_2\widehat{\nabla}_{\theta_T}\mathcal{L}_{\text{dyn}}
$$

차이는 야코비안의 비대각 블록에 있다.

| 모드 | $\partial_{\theta_T}\nabla_\theta\mathcal{L}_{\text{tone}}$ | $\partial_\theta\nabla_{\theta_T}\mathcal{L}_{\text{dyn}}$ | 구조 |
|---|---|---|---|
| split | $\mathbf{0}$ ($\mathcal{L}_{\text{tone}}$ 이 $\theta_T$ 와 무관) | $\neq\mathbf{0}$ | **하삼각** — $\theta$ 동역학이 자율적 |
| **selective** | $\neq\mathbf{0}$ | $\neq\mathbf{0}$ | **2인 게임** — 각자 자기 손실만 내려가되 지형은 서로 바꾼다 |
| unified | $\neq\mathbf{0}$ | $\neq\mathbf{0}$ | 게임 + 그래디언트 공유 |

`selective` 에서 값이 결합되는 이유: $\mathcal{L}_{\text{tone}}$ 은 컴프를 통과한 신호에서 재므로 **$\theta_T$ 가 바뀌면 손실 값이 바뀐다.** 그래디언트만 막혔을 뿐 지형은 공유한다. 따라서 **"$\theta$ 의 동역학이 자율적"이라는 명제는 성립하지 않고**, 수렴은 구조가 아니라 실측으로 확인해야 한다.

고정점은 각자의 최적 응답이 동시에 성립하는 점이다:

$$\nabla_\theta\mathcal{L}_{\text{tone}}(\theta^\ast,\theta_T^\ast)=0,\qquad \nabla_{\theta_T}\mathcal{L}_{\text{dyn}}(\theta^\ast,\theta_T^\ast)=0$$

> **진단상의 함의**(세 모드 공통): 단일 목적함수가 없으므로 **"총손실 단조 감소"라는 판정 기준이 성립하지 않는다.** 수렴은 두 손실을 각각 봐야 한다.

---

## 11-D. 세 모드 실측 — 왜 selective 인가

같은 소재, 50스텝. (이 표는 $L_{\text{dyn}}$ 이 아직 PLR 이던 시점의 측정이다 — 모드 비교가 목적이므로 지표는 통제 변수다.)

**A. `mixpractice vox` → `Golden Acapella`**

| 모드 | tonal | dyn | match | $T$ | $\text{GR}_{\max}$ | $\gamma$ 범위 |
|---|---|---|---|---|---|---|
| **selective** | **2.138** | 0.523 | **1.573** | $-26.9$ | $-16.85$ | $-4.4\sim+3.7$ |
| unified | 2.593 | **0.137** | 1.733 | $-29.2$ | $-16.62$ | $-4.9\sim+5.4$ |

**B. `은주막걸리 vox` → `Effie MAKGEOLLI BANGER` (Demucs 분리)** — 실전 실패 재현 소재

| 모드 | tonal | dyn | match | $T$ | $\text{GR}_{\max}$ | EQ 커브 | **저역 최저** |
|---|---|---|---|---|---|---|---|
| **selective** | **2.383** | **5.577** | **3.501** | $-38.9$ | $-22.36$ | $-8.2\sim+10.8$ | $\mathbf{-7.6}$ dB |
| unified | 2.543 | 6.496 | 3.927 | $-37.8$ | $-12.85$ | $-21.0\sim+0.0$ | $\mathbf{-19.9}$ dB |

**unified 의 실패**: EQ 가 저역을 $-19.9$ dB 까지 파낸다. $L_{\text{dyn}}$ 이 $\theta$ 로도 흐르기 때문에, EQ 가 스펙트럼을 깎아 다이내믹 수치를 대신 맞추려 든 결과다. `selective` 로 경로를 끊자 $-7.6$ dB 로 정상 범위에 들어오고 세 지표가 모두 개선됐다.

**남은 문제**: B 소재의 PLR 역주행($13.89\to16.33$, 목표 $10.75$)은 `selective` 로도 재현됐다. **그래디언트 결합이 원인이 아니라는 뜻**이고, 여기서부터 §22-B → §15 의 진단 사슬이 시작된다.

---

## 11-E. 측정 지점 — 리버브 앞인가 뒤인가

`LOSS_GRAD_MODE` 가 **그래디언트 경로**를 정한다면, `LOSS_MEASURE_POINT` 는 **체인 어디서 재는가**를 정한다. 직교하는 축이다.

| | `"pre_reverb"` (예전) | `"post_reverb"` (기본) |
|---|---|---|
| 측정 지점 | 컴프 출력 | **체인 최종 출력** |
| 리버브의 역할 | 렌더 전용 후처리 | 손실 경로의 일부 |
| $\partial L_{\text{tone}}/\partial\theta$ | 리버브를 안 지남 | 리버브를 **관통** |

**왜 바꿨나.** 예전에는 손실이 최소화하는 양과 화면에 찍히는 양이 다른 신호였다. 지표는 언제나 최종 출력에서 잰다(`y_processed.mean(axis=0)`) — 리버브·`match_volume`·소프트 리미터를 모두 거친 뒤다. 학습은 그 앞에서 끝나 있었다. §18 감사 A 와 같은 종류의 불일치다.

렌더와 학습이 같은 `E2EChain.apply_reverb` 를 호출하므로, 지금은 이 동일성이 **정의상** 보장된다.

**실측** (50스텝, 측정 IR·반주 기준):

| | A. `mixpractice vox`→`Golden` ||| B. `은주막걸리`→`MAKGEOLLI BANGER` |||
|---|---|---|---|---|---|---|
| | pre | post | Δ | pre | post | Δ |
| tonal | 3.389 | **3.031** | $-0.358$ | 2.636 | **2.263** | $-0.373$ |
| dyn_err | 1.837 | **1.642** | $-0.195$ | 14.899 | **14.659** | $-0.240$ |
| match | 2.846 | **2.545** | $-0.301$ | 6.928 | **6.602** | $-0.326$ |
| $T$ | $-39.8$ | $-18.2$ | | $-44.5$ | $-44.5$ | |
| $\text{GR}_{\max}$ | $-25.4$ | $-10.1$ | | $-25.5$ | $-25.4$ | |
| EQ 최저 $\gamma$ | $-3.6$ | $-6.2$ | | $-6.1$ | $-7.6$ | |
| 학습 시간 | 38.4 s | 81.5 s | **2.12×** | 45.7 s | 89.7 s | **1.96×** |

지표 8건 전부 개선. **저역 붕괴 없음** — 최저 $-7.6$ dB 로 §11-D 의 unified 실패($-19.9$ dB)와는 다른 영역이다. 대신 **저역 컷 + 고역 부스트 틸트**가 일관되게 강해졌다(A 의 10 kHz: $+0.04\to+5.92$ dB). 리버브가 저중역을 채우고 고역을 롤오프하니 EQ 가 그 역함수를 만드는 것으로, 방향은 물리적으로 맞다.

**컴프 거동이 A 에서 크게 바뀌었다**: $T$ $-39.8\to-18.2$, $\text{GR}_{\max}$ $-25.4\to-10.1$ dB. 리버브 테일이 RMS 를 올려 CF 를 낮추므로 목표 CF 에 닿는 데 압축이 훨씬 덜 필요해진다. 청감 확인 결과 채택. B 는 이미 threshold 바닥($R=3$ 한계)이라 변화 없음.

**비용**: 스텝마다 곡 전체 FFT 컨볼루션이 tone·dyn 각각 하나씩 backward 까지 붙는다(A 기준 스텝당 $0.58\to1.42$ s, $+0.84$ s). "리버브 4조합 모두 렌더"는 `match_e2e` 를 통째로 4번 부르는 옵션이라 이 배수가 그대로 곱해진다.

### 남은 비대칭 — "웻+웻 대 웻"

방향은 맞다. 예전에는 *드라이 내 보컬* vs *웻 레퍼런스* 를 비교했고, EQ 가 레퍼런스 잔향이 만든 스펙트럼 차이를 게인으로 흉내낸 뒤 렌더에서 리버브가 또 얹혀 **이중 계상**됐다. 그 왜곡이 빠진 몫이 위 tonal 개선이다.

그래도 정확히 "웻 대 웻"은 아니다:

1. **내 보컬의 원래 잔향이 남아 있다.** 우리 쪽은 `내 방 잔향 + 레퍼런스 IR`, 타깃은 `레퍼런스 잔향` 하나 — 실제로는 **"웻+웻 대 웻"**이다. 잔향이 과대 표현되므로 EQ 가 저역을 필요보다 더 깎는 쪽으로 편향되고, 위 표의 저역 컷 심화 중 어디까지가 정당한 보정이고 어디부터가 이 편향인지 구분이 안 된다. 근본 해결은 `ref_dry`(레퍼런스 디리버브)이고 미구현이다.
2. **IR 출처가 반주다**(기본 `ir_source="instrumental"`). 반주 잔향 ≠ 레퍼런스 보컬 잔향. 더하는 웻이 타깃의 웻과 같은 공간이라는 보장이 없다.
3. **EQ 커브가 리버브 설정에 종속된다.** `reverb_amount`·RT60·wet 을 바꾸면 최적 EQ 가 달라진다. 최종 출력 기준이라는 점에서 옳지만, EQ 결과를 리버브 설정 간에 재사용할 수 없다.

-->
---

## 12. $L_{\text{tone}}$ — 정의 (현재)

입력은 **체인 최종 출력** $\text{tone}_{\text{src}}=\text{Rev}\big(\text{Comp}(y_{\text{eq}};\,\text{sg}[\theta_T])\big)$ 다 — EQ→컴프→리버브를 모두 통과하고, 컴프 파라미터 $\theta_T$ 만 detach 로 상수화된다: $P(f,t)=\lvert\text{STFT}(\text{tone}_{\text{src}})\rvert^2$.

$$\bar P(f)=\frac1N\sum_t P(f,t),\qquad M(m)=\sum_f\Psi(m,f)\bar P(f),\qquad D(m)=10\log_{10}(M(m)+\epsilon)$$

**레벨 정렬 — 평가 대역 $\Omega$ 안의 평균 dB 제거**:

$$\boxed{\ \tilde D(m)=D(m)-\frac{1}{\lvert\Omega\rvert}\sum_{m'\in\Omega}D(m')\ }$$

$$L_{\text{tone}}=\frac{1}{\lvert\Omega\rvert}\sum_{m\in\Omega}\big\lvert \tilde D_{\text{out}}(m)-\tilde D_{\text{tgt}}(m)\big\rvert$$

---

**레벨 불변성**: $y\to cy \Rightarrow M\to c^2M \Rightarrow D\to D+20\log_{10}c$ (모든 $m$ 공통) $\Rightarrow$ 평균도 같은 양 이동 $\Rightarrow \tilde D$ 불변. $\square$

**$L_1$ 을 쓰는 이유**: mel 밴드 하나의 큰 이상치(예: 기음 불일치)가 $L_2$ 처럼 제곱으로 증폭되지 않는다. 다만 $L_1$ 의 최적 오프셋은 **중앙값**인데 여기서는 평균을 뺀다 — 즉 $L_{\text{tone}}$ 은 오프셋에 대해 완전 최적화된 값보다 항상 크거나 같다(상계).

---

<!--
## 13. $L_{\text{tone}}$ — 이전 정의와 민감도 비교

**이전 정의** (총에너지 나눗셈):

$$D^{\text{old}}(m)=10\log_{10}\frac{M(m)}{\sum_{m'}M(m')}=D(m)-\Lambda,\qquad \Lambda=10\log_{10}\sum_{m'}M(m')$$

합 $\sum_{m'}$ 은 $\Omega$ 밖(대역 하한 미만 + 상한 이상)까지 포함한다.

**밴드 $i$ 에 $+\Delta$ dB 를 가했을 때 잔차 $r_m$ 의 변화**:

$$\frac{\partial r^{\text{old}}_m}{\partial\Delta}=\delta_{mi}-w_i,\qquad w_i=\frac{M(i)}{\sum_{m'}M(m')}$$

$$\frac{\partial r^{\text{new}}_m}{\partial\Delta}=\delta_{mi}-\frac{\mathbb{1}[i\in\Omega]}{\lvert\Omega\rvert}$$

*(유도: $\frac{d}{d\Delta}10\log_{10}\sum M = \frac{M_i}{\sum M}=w_i$)*

**해석**: 보컬은 저역이 에너지를 지배하므로 저역 밴드의 $w_i$ 가 크다. 이전 정의에서는 그 밴드를 부스트하면 자기 잔차는 $(1-w_i)\Delta$ 만 늘고 **나머지 모든 밴드가 $-w_i\Delta$ 씩 줄어든다.** 평가 밴드가 수십 개이므로 총합에서는 이득이 될 수 있다 — "다른 대역이 과하다"를 그 대역을 깎는 대신 **저역을 키워 상대 비중을 낮추는 해**가 성립한다.

새 정의에서는 결합이 $1/\lvert\Omega\rvert\approx 0.013$ 으로 사실상 소멸한다.

---

-->

## 14. LUFS (ITU-R BS.1770-4)


K-weighting $\mathcal{K}$ = 2단 biquad (`pyloudnorm` 계수를 그대로 torch `lfilter` 로 적용):

$$z_j=\frac1L\sum_{n\in\text{block}_j}\big(\mathcal{K}\{y\}[n]\big)^2,\qquad \ell_j=-0.691+10\log_{10}z_j$$

블록 $L=400$ ms, 홉 $100$ ms (75 % 오버랩).

**2단 게이팅** (계단 대신 폭 1 dB 시그모이드 — 미분 가능성 확보):

$$g^{\text{abs}}_j=\sigma(\ell_j+70)$$

$$\Gamma=\Big[-0.691+10\log_{10}\frac{\sum_j g^{\text{abs}}_j z_j}{\sum_j g^{\text{abs}}_j}\Big]-10\quad(\text{detach})$$

$$g_j=g^{\text{abs}}_j\cdot\sigma(\ell_j-\Gamma),\qquad \text{LUFS}=-0.691+10\log_{10}\frac{\sum_j g_jz_j}{\sum_j g_j}$$

$\Gamma$ 를 detach 하는 이유: $\Gamma$ 자체를 미분하면 게이트가 스스로를 이동시키는 이차 효과가 생겨 학습이 불안정해진다.

---

## 15. $L_{\text{dyn}}$ — True Peak, Crest Factor, 불변성

**True Peak** (4배 오버샘플 $u$, straight-through):

$$\text{TP}_{\text{dB}}=20\log_{10}\Big[\underbrace{\max_n\lvert u[n]\rvert}_{\text{forward 값}}+\underbrace{\big(\overline{\text{top-}k}-\text{sg}[\overline{\text{top-}k}]\big)}_{\text{backward 경로}}\Big],\qquad k=64$$

값은 정의 그대로 정확하고, 그래디언트는 상위 $k$ 개 표본에 분산된다(하드 $\max$ 는 표본 하나에만 흘러 불안정).

**RMS** — K-weighting 도 게이팅도 없는 단순 실효값:

$$\text{RMS}_{\text{dB}}=20\log_{10}\sqrt{\tfrac1N\textstyle\sum_n y[n]^2}$$

$$\text{CF}=\text{TP}_{\text{dB}}-\text{RMS}_{\text{dB}}
\qquad\Longrightarrow\qquad
\boxed{\ L_{\text{dyn}}=\big\lvert\text{CF}_{\text{out}}-\text{CF}_{\text{tgt}}\big\rvert\ }$$

---

**레벨 불변성**: $y\to cy \Rightarrow \text{TP}_{\text{dB}},\ \text{RMS}_{\text{dB}}$ 가 모두 $+20\log_{10}c$ $\Rightarrow$ CF 불변. $\square$ (실측: $\times0.25/\times1/\times2$ 에서 CF $=13.5161$ 로 동일)

### 왜 PLR 에서 크레스트 팩터로 바꿨나

이전 정의는 $\text{PLR}=\text{TP}_{\text{dB}}-\text{LUFS}$ 였다. 분모의 LUFS 가 K-weighting + 2단 게이팅(절대 $-70$ LUFS, 상대 $-10$ LU)을 거치는데, **그 게이팅이 압축 깊이에 따라 TP 와 다른 속도로 움직인다.** 

$$\Delta\text{TP}=-17.8\ \text{dB},\qquad \Delta\text{LUFS}=-22.1\ \text{dB}\qquad\Rightarrow\qquad \Delta\text{PLR}=+4.4\ \text{dB}$$

---

## 16. 정규화항 — 남은 것과 사라진 것

**EQ 쪽** ($\mathcal{L}_{\text{tone}}$ 에 포함, 밴드 게인 $\gamma$ 에 대해):

$$R_{\text{eq}}=\lambda_2\frac1J\sum_j\gamma_j^2+\lambda_s\frac{1}{J-1}\sum_j(\gamma_{j+1}-\gamma_j)^2$$

하드 클램프 $G_{\max}\tanh\theta$ 는 상한 도달 전까지 저항이 $0$ 이라 사후 절단과 같아진다. 제곱 페널티는 한계비용이 게인에 비례한다: $\partial R/\partial\gamma_j = 2\lambda_2\gamma_j/J$.

**컴프 쪽** ($\mathcal{L}_{\text{dyn}}$ 에 포함):

$$w_t\Big(\frac{T}{-60}\Big)^2,\qquad w_t=0\ (\text{기본 비활성})$$

$T$ 를 내리는 쪽은 그래디언트 지렛대가 길다(활성 프레임 수까지 함께 늘어난다). 목표 PLR 을 $T$ 바닥으로 맞춰 버리면 전 구간이 threshold 위라 **컴프가 아니라 포락선 스케일러**가 된다. ratio 고정 이후 $T$ 혼자 압축량을 감당하므로 이 위험이 커졌다.

### 삭제된 항

| 삭제 | 이유 |
|---|---|
| $w_a e^{-(R-1)}$ (bypass 방지) | ratio 를 상수로 고정해 퇴화 방향 자체가 사라짐 (§7) |
| auto_balance $\ \tilde L_i = L_i/\hat e_i$ | 손실을 더하지 않으므로 단위를 맞출 이유가 없음 (아래 불변성) |
| $w_1, w_2$ | 손실이 분리된 뒤에는 각 손실의 상수배일 뿐 (아래 불변성) |

### 왜 상수배가 무의미한가 — Adam 의 스케일 불변성

손실을 $c>0$ 배 하면 그래디언트도 $c$ 배가 되고, Adam 의 1·2차 모멘트가 각각 $c$, $c^2$ 배 되므로

$$
\Delta\theta=-\eta\frac{\hat m}{\sqrt{\hat v}+\epsilon}
\;\xrightarrow{\;L\to cL\;}\;
-\eta\frac{c\,\hat m}{c\sqrt{\hat v}+\epsilon}\;\approx\;\Delta\theta
\qquad(\epsilon \ll c\sqrt{\hat v})
$$

즉 **가중치 $w_i$ 나 EMA 정규화 $1/\hat e_i$ 는 갱신량을 바꾸지 못한다.** 손실이 하나로 합쳐져 있을 때는 $w_i$ 가 항들의 **상대** 비중을 정해서 의미가 있었지만, 분리된 뒤에는 각자가 자기 손실의 상수배라 효과가 사라진다. 학습 속도는 $\eta$ 로만 조절된다.

> 남은 두 정규화는 **모듈 자신의 파라미터 제약**이지 모듈 간 침범 방지가 아니다. 침범 방지는 이제 정규화가 아니라 **그래프 구조**가 담당한다.

---

## 17. 보고 지표 정의

$$
\text{tonal\_error}=\frac{1}{\lvert\Omega\rvert}\sum_{m\in\Omega}\lvert \tilde D_{\text{proc}}(m)-\tilde D_{\text{ref}}(m)\rvert
$$

$$
\text{dynamics\_error}=\lvert \text{CF}_{\text{proc}}-\text{CF}_{\text{ref}}\rvert,\qquad
\text{match\_error}=0.65\,\text{tonal}+0.35\,\text{dyn}
$$

지표는 크레스트 팩터 하나로 통일됐다 — CF 는 피크 $\ge$ RMS 라 항상 $\ge0$ 이므로, PLR 시절의 "$\text{PLR}_{\text{ref}}\le0$ 이면 대체" 예외가 필요 없다.

$$
\text{GR}_{\max}=\min_k G_k\Big|_{\text{EQ}(\alpha)\ \text{적용된 신호에 컴프 재통과}}
$$

**측정 시점**: 세 손실의 원시값($L_{\text{tone}}, L_{\text{dyn}}, L_{\text{decay}}$)은 학습 궤적이 아니라 **최종 렌더 출력에서 1회** 계산한다(`DSPMatchingLoss.report`). 학습 루프가 계산하지 않은 항도 여기서는 값이 나온다.

**컴프 파라미터 표시**: $R$ 은 항상 $3.0$ 으로 고정 표시되고, 학습 결과로 움직이는 것은 $T$ 와 $\text{GR}_{\max}$ 뿐이다.

**렌더 후처리** (손실이 보지 못하는 연산):

- `match_volume` (옵션): $\mathbf{y}\leftarrow \mathbf{y}\cdot\dfrac{\text{RMS}(r)}{\text{RMS}(\mathbf{y})}$
- 소프트 리미터: 피크 $>0.88$ 일 때 $\lvert y\rvert>0.85$ 인 표본에만
$$\hat y=\operatorname{sign}(y)\Big[0.85+0.11\tanh\frac{\lvert y\rvert-0.85}{0.11}\Big]$$


<!--
---

## 18. 감사 A — 학습과 렌더의 파라미터 불일치

**현상**: 학습 루프는 `chain(...)` 을 거치지 않고 `eq_output` → `comp` → `apply_reverb` 를 직접 조립하는데(`pipeline.py` 의 tone_src/dyn_src 생성부), 거기서 $\alpha,\rho$ 를 넘기지 않아 $\alpha=1,\ \rho=1$ 이다. ($\text{reverb\_amount}$ 만은 렌더와 같은 값이 들어간다.)
렌더는 `chain(t_raw, eq_amount=α, comp_amount=ρ)` → UI 기본 $\alpha=0.8$.

$$\theta^\ast=\arg\min_\theta \mathcal{L}\big(\text{출력}(\theta;\alpha=1)\big)\qquad\text{인데 실제 출력은}\qquad \text{출력}(\theta^\ast;\alpha=0.8)$$

**왜 문제인가**: $\mathcal{L}$ 은 $\alpha$ 에 대해 볼록하지도 선형이지도 않다. 특히 컴프는 EQ 출력 레벨에 비선형으로 반응하므로($E_k$ 가 바뀌면 $o_k$ 가 바뀜), $\alpha$ 를 줄이면 컴프 동작점 자체가 학습 시점과 달라진다.

**확인 방법**: 같은 입력으로 $\alpha=1.0$ 과 $0.8$ 렌더의 `tonal_error` 를 비교. $\alpha=0.8$ 이 더 크면 최적점에서 벗어난 것이다.

**수정 후보**
1. 학습에도 $\alpha,\rho$ 를 그대로 넣어 목적함수와 출력을 일치시킨다.
2. 또는 $\alpha$ 를 "학습 후 곡선 스케일링"이라는 별도 의미로 문서화하고 지표는 $\alpha$ 적용 후 값으로만 보고한다.

---

## 19. 감사 B — 고정 HPF/LPF 의 비대칭과 차수 *(해결됨 — 필터 삭제)*

> **상태: 해결.** HPF·LPF 를 코드에서 삭제했다(§5). 아래는 왜 문제였는지의 기록이다.

**현상 1 (비대칭)**: $M_{\text{filt}}$ 는 **출력에만** 곱해진다. 레퍼런스 $r$ 에는 없다. 그런데 $L_{\text{tone}}$ 은 둘의 mel dB 를 비교한다.

$$\tilde D_{\text{out}} \ \text{는 }\ 16\ \text{kHz 이상에서 감쇠된 값},\qquad \tilde D_{\text{tgt}}\ \text{는 감쇠 없음}$$

$\Rightarrow$ 손실은 이 차이를 "톤 부족"으로 읽고 **EQ 가 고역을 밀어 올려 LPF 를 상쇄하려 한다.**
실측(변경 전 코드): $8$–$20$ kHz 구간 적용 EQ $= +4.82$ dB.

**현상 2 (차수)**: 버터워스 **진폭** 응답은 $\lvert H\rvert=\big(1+(f/f_c)^{2p}\big)^{-1/2}$ 인데 코드는

$$M_{\text{filt}} = \frac{1}{1+(f/f_c)^{2p}} = \lvert H\rvert^2$$

를 진폭 배수로 쓴다. 즉 dB 감쇠가 **의도의 2배**(2차인데 $-24$ dB/oct 로 동작). 확인: $f=f_c$ 에서 $-6.02$ dB (버터워스 정의는 $-3.01$ dB).

**수정 후보**: (i) 타깃에도 같은 $M_{\text{filt}}$ 를 곱해 비대칭 제거, (ii) $M_{\text{filt}}\leftarrow\sqrt{\cdot}$ 로 차수 정정, (iii) 필터를 손실 밖(렌더 전용)으로 이동.

---

## 20. 감사 C — 대역 경계 마스크 *(부분 해결)*

**현상**: $M_{\text{eq}}$ 는 계단인데 밴드는 폭 $\sigma$ 의 가우시안이라 경계에서 커브가 잘린다. 밴드 하한을 $200$ Hz 로 올린 뒤에도 **최저 밴드($f_0=200$ Hz)의 아래쪽 절반은 여전히 마스크에 잘린다**:

$$\gamma_0\ \text{에 }+G\ \text{를 주면}\quad G_{\text{dB}}(f)=G\,e^{-(\log_2 f-\log_2 200)^2/2\sigma^2}\ \text{인데}\ f<200\ \text{에서 }0\ \text{으로 강제}$$

하한 바로 아래는 $0$ dB, 바로 위는 $\gamma_0$ 의 가우시안 값 — 경계에서 불연속이다.

**남은 문제 (해결 안 됨) — 경계에서 손실이 눈을 반쯤 감는다.**

$\Omega$ 는 mel 밴드 **중심**이 $[200, 10000)$ Hz 인 밴드로 정의된다($\lvert\Omega\rvert=61$). slaney mel 은 1 kHz 미만이 선형 간격이고($n_{\text{mels}}=80$ 기준 실측 $49.3$ Hz), 하한에서 제외되는 밴드는 중심 $50.4 / 98.3 / 147.8 / 198.0$ Hz 네 개다. 문제는 마지막 밴드($198.0$ Hz)의 삼각 필터가 $150.7$–$236.9$ Hz 에 걸쳐 **$200$ Hz 위까지 넘어온다**는 점이다. bin 별로 "손실이 실제로 보는 가중치 비율"을 재면:

| bin | 193.8 Hz | 215.3 Hz | 236.9 Hz | 258.4 Hz |
|---|---|---|---|---|
| 손실이 보는 비중 | $0\%$ | $\mathbf{36\%}$ | $\mathbf{80\%}$ | $100\%$ |

즉 $200$–$260$ Hz 는 **부분적으로만** 평가된다. EQ 최저 밴드($f_0=200$ Hz)가 바로 이 구간에 중심을 두므로, $\gamma_0$ 은 자기 영향 범위의 일부만 손실에 반영된 채 학습된다.

**해결된 부분**: $100$ Hz 미만에 중심을 둔 EQ 밴드 7개가 사라져, 마스크 아래에서 그래디언트 없이 표류하던 파라미터가 없어졌다.

**상한 쪽도 같은 구조**: $10$ kHz 경계에서도 $M_{\text{eq}}$ 가 계단으로 자르고 최상위 밴드($f_{16}=10$ kHz)의 위쪽 절반이 잘린다. 다만 그 위는 애초에 신뢰할 수 없는 대역이라 실익이 없다.

**남은 수정 후보**: (i) $M_{\text{eq}}$ 를 계단 대신 부드러운 롤오프로, (ii) 평가 대역을 EQ 대역보다 **넓게** 잡아 경계 스필을 손실이 보게 하기, (iii) $\Omega$ 를 "중심" 기준이 아니라 "지지 구간" 기준으로 재정의.

---

## 21. 감사 D — 정규화항의 '반쯤 죽은' 밴드 *(해결됨)*

**이전 상태**: $J=30$, 하한 $20$ Hz 에서 $f_j<100$ 인 밴드 7개는 커브의 $100$ Hz 미만 부분만 버려지고 위쪽 꼬리는 살아 있었다. 정규화항은 그 밴드들을 동등하게 취급했다:

$$R_{\text{eq}}=\lambda_2\frac1J\sum_{j}\gamma_j^2+\lambda_s\frac{1}{J-1}\sum_j(\gamma_{j+1}-\gamma_j)^2$$

$$\frac{\partial R_{\text{eq}}}{\partial\gamma_6}= \frac{2\lambda_2\gamma_6}{J}+\frac{2\lambda_s}{J-1}\big(2\gamma_6-\gamma_5-\gamma_7\big)$$

$\gamma_6$(84 Hz)은 손실로부터 약한 신호만 받으므로 이 식이 지배했고, **데이터 항이 없는 밴드가 평활항을 통해 살아 있는 밴드($\gamma_7$)를 끌어당겼다.**

**해결**: 밴드 하한을 $100$ Hz 로, 이어서 $200$ Hz 로 올려 문제의 밴드들을 제거했다. 이제 모든 $\gamma_j$ 가 평가 대역 $\Omega$ 안에 중심을 두므로 정규화항의 모든 항이 데이터 신호를 받는 파라미터에 걸린다.

$$J:\ 30\to23\to\mathbf{17},\qquad \#\{j: f_j<f_{\min}\}:\ 7\to 0\qquad(f_{\min}:\ 20\to100\to\mathbf{200}\ \text{Hz})$$

> 남은 미세한 비대칭: 최저 밴드 $\gamma_0$ 는 가우시안 아래쪽 절반이 마스크에 잘리고, 그 위 $200$–$260$ Hz 도 손실에 $36$–$80\%$ 만 반영된다(§20). 즉 유효 기여가 다른 밴드보다 작다. 밴드별 유효 가중 $\nu_j=\sum_f\Phi(f,j)M_{\text{eq}}(f)\big/\sum_f\Phi(f,j)$ 로 $R_{\text{eq}}$ 를 정규화하면 완전히 없앨 수 있으나, 밴드 1개의 문제라 현재는 두었다.

---

## 22. 감사 E — 그 밖의 확인 필요 지점

| # | 지점 | 내용 | 영향 |
|---|---|---|---|
| E1 | $\Omega$ 정의 불일치 | 손실은 **필터뱅크 무게중심**, 보고 지표는 `librosa.mel_frequencies` **중심주파수** 기준(둘 다 $[200,10000)$) | 경계 밴드 1개 차이 가능 → 손실과 지표가 미세하게 다른 집합 |
| E2 | 손실 밖 후처리 | 소프트 리미터·`match_volume` 은 학습이 못 본다 | 리미터가 피크를 깎으면 실제 PLR 이 학습값과 달라짐 |
| E3 | 절대 게이트 | $\sigma(\ell+70)$ 는 절대 레벨 기준 → PLR 의 스케일 불변성이 엄밀히는 근사 | 매우 조용한 신호에서만 문제 |
| E4 | 메이크업 활성집합 | $\mathcal{A}$ 가 $\max_k E_k$ 에 의존 → EQ 가 바뀌면 $\mathcal{A}$ 도 바뀜 | 컴프↔EQ 약한 결합 |
| E5 | ~~API 기본 밴드 수~~ | **해결**: `main.py` / `pipeline` / UI 모두 $J=17$ 로 통일 | — |
| E6 | 잔여 지표 | `compression_data` 의 crest factor·rms_var 은 더 이상 최적화 대상이 아님 | 화면 해석 혼란 |
| E7 | ~~ratio 고정값~~ | **해결**: $R$ 을 $[1,1000]$ 설정 상수로 노출했다(`COMP_RATIO`, UI 슬라이더 1–30). CF 지표에서는 $R$ 이 클수록 도달 가능 범위가 넓어진다 — 실측 $R{=}3$ 은 $T{=}-50$ 에서 CF 17.71 이 한계지만 $R{=}30$ 은 10.47 까지 내려가 목표(12.16)를 지난다 | 소재별 적정값은 사용자가 고른다(장르에 따라 3–30) |
| E8 | $L_1$ 오프셋 | 평균을 빼지만 $L_1$ 최적 오프셋은 중앙값 | 손실이 상계로 계산됨 |
| E9 | ~~톤 측정 지점~~ | **재평가 완료**: 톤을 컴프 **이후**($y_{\text{full}}$)에서 재도록 바꿨다. 실측 결과 톤은 $2.553\to2.593$ ($1.6\%$ 악화)에 그치고 다이내믹은 $0.455\to0.137$ 개선 → 종합 개선. 시변 게인 잔차는 남지만 크기가 확인됐다 | 해소 |
| E10 | ~~EQ 저역 붕괴~~ | **해결**: `unified` 에서 EQ 가 저역을 $-19.9$ dB 까지 파내던 문제. $L_{\text{dyn}}$ 이 $\theta$ 로도 흘러 EQ 가 스펙트럼을 깎아 다이내믹을 대신 맞춘 것이 원인이었고, `selective` 로 경로를 끊어 $-7.6$ dB 로 정상화 | 해소 |

---

## 22-B. 감사 F — 컴프 하이라이트 구간 전환과 $\sigma^{\text{ST}}$ 표본 부족

> **상태: 해결(제거)** — $\sigma^{\text{ST}}$ 보조항을 **완전히 제거**했다(`USE_ST_TERM=False`). 아래는 왜 제거했는지의 기록이며, 재도입 검토는 후속 과제로 남는다.

**배경**: 컴프의 학습·평가 대상을 곡 전체에서 **하이라이트 15초 구간**으로 옮겼다. 실제 믹싱에서 후렴 같은 가장 큰 대목을 기준으로 컴프를 잡는 것과 같은 정의다. 구간은 `loudest_window`(BS.1770 K-weighting 슬라이딩)로 내 보컬·레퍼런스에서 **각각 독립적으로** 고른다 — 다른 연주라 시간 정렬이 성립하지 않고, PLR 은 통계량이라 정렬이 필요 없다.

| | 학습·평가 축 | $T$ | $\text{GR}_{\max}$ | $\text{PLR}_{\text{tgt}}$ | $\text{PLR}_{\text{out}}$ | dyn_err | 시간 |
|---|---|---|---|---|---|---|---|
| 이전 | 곡 전체 | $-14.0$ dB | $-6.4$ dB | 12.00 | 11.89 | 0.116 | 56.8 s |
| **현재** | **하이라이트 15 s** | $-27.5$ dB | $-15.4$ dB | 10.37 | 10.83 | **0.455** | 32.8 s |

학습 축과 판정 축을 일치시킨 값이 $0.455$ 다(같은 결과를 곡 전체 축으로 재면 $0.655$ — PLR 은 게이팅 통계라 구간이 바뀌면 값 자체가 이동하고 그 이동량이 신호마다 달라, 두 축은 비교 대상이 아니다).

$T=-27.5$ dB 는 타당하다. 구간에서 $13.10 \to 10.37$ 로 **$2.73$ dB** 를 줄여야 하는데 $R$ 이 $3{:}1$ 고정이므로 threshold 를 그만큼 내려야 하고, 실제로 $2.27$ dB(요구량의 $83\%$)를 달성했다.

---

## 22-B (계속). 잔여 오차 $0.455$ dB 의 분해와 원인

**현상** — 하이라이트 구간에서 $L_{\text{dyn}}$ 의 두 항을 분리해 재면:

| | PLR (dB) | 단기 LUFS std |
|---|---|---|
| raw | 13.10 | 1.06 |
| ref (목표) | 10.37 | 0.34 |
| 출력 | 10.83 | 0.86 |

$$\lvert\Delta\text{PLR}\rvert = 0.454,\qquad \tfrac12\lvert\Delta\sigma^{\text{ST}}\rvert = 0.258$$

**원인** — $\sigma^{\text{ST}}$ 는 3 s 창·1 s 홉의 표준편차다. 구간을 자르면 표본 수가 급감한다:

$$
M_{\text{ST}} = \left\lfloor \frac{T_{\text{seg}} - 3}{1} \right\rfloor + 1
\quad\Longrightarrow\quad
\underbrace{126}_{128\ \text{s 곡 전체}} \;\longrightarrow\; \underbrace{13}_{15\ \text{s 구간}}
$$

표본이 $1/10$ 로 줄어 추정이 불안정해지고, 그 잡음이 threshold 를 흔든다. 레퍼런스 하이라이트가 $\sigma^{\text{ST}}=0.34$ 로 매우 균일해(이미 컴프가 걸린 상용 아카펠라의 후렴) 보조항이 요구하는 방향과 PLR 항이 요구하는 방향이 어긋나는 것도 겹친다.

**확인 방법** — 학습 로그의 threshold 궤적. 압축을 **줄이는** 방향으로 되돌아간다:

$$T:\ -30.3 \;\to\; -27.3 \;\to\; -27.5\ \text{dB}\qquad (\text{step } 0 \to 25 \to 49)$$

PLR 항 단독이라면 계속 내려가야 하는데 올라왔다 — 보조항이 반대로 당긴 결과다.

**수정 후보**

| # | 방법 | 효과 | 부작용 |
|---|---|---|---|
| 1 | $\sigma^{\text{ST}}$ 창 축소 (3 s → 1 s, 홉 0.25 s) | 표본 $13 \to 57$ | 창이 짧아져 '단기 라우드니스'의 의미가 달라짐 |
| 2 | 보조항 가중치 하향 ($0.5 \to 0.1$) | PLR 단독 매칭에 집중 | $(T)$ 를 구속하는 정보가 줄어듦 |
| 3 | 보조항을 **분위수 곡선**으로 교체 | 표본 수 민감도 자체를 제거 | 구현 비용, §20 향후 과제와 통합 필요 |

### 실제로 한 조치 — 보조항 제거

위 세 후보 중 어느 것도 아니라 **항 자체를 없앴다.** 다른 소재(`은주막걸리 vox` → `Effie MAKGEOLLI BANGER`)에서 이 항의 해악이 표본 부족보다 훨씬 심각하게 드러났기 때문이다:

$$\sigma^{\text{ST}}_{\text{raw}} = 8.22 \quad\text{vs}\quad \sigma^{\text{ST}}_{\text{ref}} = 0.71\qquad(\text{12배 격차})$$

보조항이 $3.3$–$5.7$, PLR 항이 $2.6$–$3.1$ 로 **보조항이 손실을 지배**했다. 옵티마이저는 그것을 줄이려 threshold 를 $-38.9$ dB 까지 밀어넣었고, 그 지점에서는 큰 대목이 전부 threshold 위라 게인이 $-29$ dB 로 사실상 고정된다 — 컴프가 아니라 **감쇠기**다. 그 결과:

| thr | TP | $\Delta$TP | LUFS | $\Delta$LUFS | PLR | $\Delta$PLR |
|---|---|---|---|---|---|---|
| $-10$ | $-8.50$ | $-2.50$ | $-21.88$ | $-1.99$ | 13.38 | $-0.51$ |
| $-30$ | $-19.51$ | $-13.51$ | $-33.31$ | $-13.42$ | 13.79 | $-0.10$ |
| $-50$ | $-26.24$ | $-20.24$ | $-42.77$ | $\mathbf{-22.88}$ | 16.54 | $\mathbf{+2.65}$ |

$\text{PLR}=\text{TP}-\text{LUFS}$ 인데 **LUFS 가 TP 보다 더 많이 떨어져** PLR 이 올라간다(목표와 반대 방향). True Peak 자체는 raw 를 넘은 적이 없고(세 조건 모두 감소), 출력 최대 피크 위치도 raw 의 최대 피크와 동일하다 — 컴프가 새 피크를 만든 것이 아니다.

**제거 후**: 손실이 $\lvert\Delta\text{PLR}\rvert$ 뿐이므로 최소점이 threshold $\approx-10$ dB 로 이동하고 깊은 쪽은 단조 증가한다. 학습된 threshold 가 $-38.9 \to -19.9$ dB 로 정상 범위에 들어왔다.

계산 코드는 `USE_ST_TERM=False` 로 남겨 두었다 — 표본 수를 늘리거나(창 축소) 분위수 곡선으로 바꾸는 재도입안은 후속 과제다.

> **후속**: $L_{\text{dyn}}$ 지표를 PLR 에서 **크레스트 팩터**로 교체하면서(§15) 이 절과 메이크업 진단이 다룬 문제들의 공통 뿌리 — **LUFS 게이팅이 압축 깊이에 따라 TP 와 다른 속도로 움직이는 것** — 이 제거됐다. CF 스윕에서는 재상승이 $+0.00$ dB 다.

---

## 22-C. 감사 G — 메이크업 게인 가설 *(기각)*

> **상태: 기각** — 가설을 세우고 실측으로 확인했으나 원인이 아니었다. 기록으로 남긴다.

**가설**: 오토 메이크업(§10)은 활성 구간 평균 GR 만큼 전체에 상수를 더한다. threshold 가 깊어지면 그 상수가 커지므로, 이것이 PLR 재상승과 "$\text{GR}_{\max}$ 는 큰데 TP·LUFS 변화는 작다"는 현상의 원인일 수 있다.

**확인 방법**: 컴프 생성자의 기존 `makeup` 플래그로 끄고 같은 스윕을 반복한다(코드 변경 불필요).

| $R$ | | 최선 PLR (그때 $T$) | $-50$ dB 까지 재상승 |
|---|---|---|---|
| 3 | 메이크업 있음 | 13.38 ($-10$) | $+3.16$ dB |
| 3 | **없음** | 13.37 ($-10$) | $+2.70$ dB |
| 30 | 메이크업 있음 | 12.77 ($-18$) | $+4.37$ dB |
| 30 | **없음** | 12.76 ($-18$) | $+3.32$ dB |

**결론**: 최소점 위치도, 최선값도 소수 둘째 자리까지 같다. 메이크업을 꺼도 재상승은 남는다(폭만 $0.5$–$1.0$ dB 감소). 메이크업은 예상대로 **상수 오프셋**으로만 작용한다 — 끄면 TP 와 LUFS 가 **둘 다** 같은 만큼 더 내려간다($R=30$, $T=-50$: TP $-34.07\to-47.26$, LUFS $-51.21\to-63.34$).

$\Rightarrow$ 원인은 다른 곳에 있다. 다음 진단이 §15 의 LUFS 게이팅이다.

---

## 23. 변경 이력 — 동치 / 비동치

**동치 (값·그래디언트가 같아야 함)**

| 변경 | 근거 |
|---|---|
| 타깃 통계 캐시 | 타깃 통계는 $r$ 만의 함수이고 $r$ 은 학습 중 상수 |
| 가중치 0 항 계산 생략 | $w_i\tilde L_i=0,\ \nabla_\theta(w_i\tilde L_i)=0$ (그 뒤 가중치 자체가 사라져, 지금은 *움직일 파라미터가 없는 손실*을 생략) |
| ratio 를 버퍼 텐서 → float 상수 | 그래프에 없던 값이라 계산에 영향 없음. 회귀 실측 소수점까지 일치 |

**비동치 — 이번 세션 (시간순)**

| # | 변경 | 이전 | 현재 | 근거 |
|---|---|---|---|---|
| 1 | 손실 그래디언트 구조 | split ($\text{sg}$, 톤은 컴프 이전) | 잠시 **unified** (결합) | 톤을 최종 출력에서 재려는 시도 |
| 2 | ↳ 되돌림 | unified | **selective** (측정은 최종 출력, 경로는 분리) | unified 에서 EQ 저역 $-19.9$ dB 붕괴 |
| 3 | 컴프 학습·평가 구간 | 곡 전체 | **하이라이트 15 s** (학습·지표 동일 구간) | §22-B |
| 4 | $L_{\text{dyn}}$ 보조항 | $+\tfrac12\lvert\Delta\sigma^{\text{ST}}\rvert$ | **제거** | 보조항이 손실을 지배해 컴프가 감쇠기로 변질 (§22-B) |
| 5 | ratio | 학습 → 상수 $3.0$ 고정 | **설정 상수** $[1,1000]$, 기본 $3.0$ | 소재별 조정 필요 (§7, E7) |
| 6 | $L_{\text{dyn}}$ 지표 | PLR $=$ TP $-$ LUFS | **CF $=$ TP $-$ RMS** | LUFS 게이팅발 비단조성 (§15) |
| 7 | 손실 측정 지점 | 컴프 출력 (리버브 전) | **체인 최종 출력** (`LOSS_MEASURE_POINT="post_reverb"`) | 손실과 보고 지표가 다른 신호를 보고 있었음. 8개 지표 전부 개선, 청감 확인 후 채택 |

**비동치 — 이전 세션 (참고)**

| 변경 | 이전 | 현재 |
|---|---|---|
| $L_{\text{tone}}$ 레벨 기준 | 총에너지 나눗셈 | 평가 대역 평균 dB 제거 (민감도 $\delta_{mi}-w_i \to \delta_{mi}-1/\lvert\Omega\rvert$) |
| 보고 지표 | 마지막 학습 스텝 | 최종 렌더 출력에서 1회 |
| EQ 밴드 범위 | $20$–$20$ k Hz, $J=30$ | $200$–$10$ k Hz, $J=17$ ($\Omega$ 와 일치) |
| 고정 HPF/LPF | $80$ Hz / $16$ kHz | 삭제 ($M_{\text{eq}}$ 가 담당) |
| bypass 벌점 · auto_balance · $w_1,w_2$ | 활성 | 삭제 / 비활성 |

**권장 검증 순서**
1. **그래디언트 격리**: $\mathcal{L}_{\text{tone}}$ backward 후 $\theta_T.\text{grad}$ 가 `None`, $\mathcal{L}_{\text{dyn}}$ backward 후 $\theta$ 그래디언트 불변 — 확인됨
2. **forward 동일성**: $\text{tone}_{\text{src}},\ \text{dyn}_{\text{src}},\ y_{\text{full}}$ 이 수치적으로 같은가 — 확인됨($0.0$). 리버브 뒤로 옮긴 뒤에는 렌더와 학습이 같은 `apply_reverb` 를 공유해 구조적으로 보장
3. **레벨 불변성**: 입력 $\times c$ → $L_{\text{tone}},L_{\text{dyn}}$ 불변 — 확인됨(CF $13.5161$ 고정)
4. §18 A: $\alpha=1$ vs $0.8$ 렌더의 `tonal_error` 비교 — 미확인
5. §19 B: 타깃에도 대역 제한을 걸었을 때 고역 부스트가 사라지는가 — 해소(필터 삭제)
6. GR 이 큰 소재에서 톤 악화가 어디까지 가는가 — 미확인


---


## 24. 요약 — 이번 세션의 진단 사슬

실전 렌더링에서 두 가지 이상이 동시에 관측됐다: **EQ 가 저역을 $-19.9$ dB 까지 파냄**, **PLR 이 목표와 반대로 상승**($13.89\to17.25$, 목표 $10.75$). 원인은 하나가 아니라 **층이 다른 세 개**였다.

```
                     증상: EQ 저역 붕괴 + PLR 역주행
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
 ① 구조적 원인             ② 파라미터적 원인          ③ 지표적 원인
 그래디언트 결합            σ_ST 보조항 지배          LUFS 게이팅 비대칭
 L_dyn → θ 로 흘러         raw 8.22 vs ref 0.71      TP 는 선형, LUFS 는
 EQ 가 스펙트럼을          → threshold −38.9 dB      더 빠르게 하락
 깎아 대신 맞춤            → 컴프가 감쇠기로          → PLR 이 U자로 반등
        │                         │                         │
        ▼                         ▼                         ▼
   selective 모드            σ_ST 항 제거              지표를 CF 로 교체
   (측정=최종 출력,          L_dyn = |ΔPLR|            L_dyn = |ΔCF|
    경로=분리)                                        TP − RMS, 게이팅 없음
        │                         │                         │
        ▼                         ▼                         ▼
   저역 −7.6 dB 로          threshold −19.9 dB        재상승 +0.00 dB
   정상화                    정상 범위 복귀             완전 단조 감소
```

**기각된 가설**: 메이크업 게인(§22-C) — 끄고 재현해도 재상승이 그대로였다(폭만 $0.5$–$1.0$ dB 감소).

**부수 변경**: ratio 를 학습 대상이 아닌 **설정 상수**로 노출($[1,1000]$, UI 슬라이더 1–30). 지표를 CF 로 바꾼 뒤로는 $R$ 이 도달 가능 범위를 실제로 넓힌다.

### 최종 검증

| 소재 | $T$ | $\text{GR}_{\max}$ | CF src → out (목표) | dyn_err | tonal | match |
|---|---|---|---|---|---|---|
| A. `mixpractice vox` → `Golden` | $-39.8$ | $-25.4$ | $14.16\to\mathbf{11.75}$ (11.10) | 0.647 | **2.134** | 1.614 |
| B. `은주막걸리` → `MAKGEOLLI BANGER` | $-44.6$ | $-26.1$ | $21.97\to\mathbf{18.70}$ (12.16) | 6.540 | 2.382 | 3.838 |

* **A(회귀 없음)**: tonal $2.134$ 로 세션 시작 시점($2.146$)과 사실상 동일. 목표 CF $11.10$ 에 $11.75$ 로 근접.
* **B(정상 수렴)**: 역주행이 사라지고 목표 방향으로 단조 이동. 남은 격차 $6.54$ 는 **지표 결함이 아니라 파라미터 범위 문제** — $R=3$ 에서는 CF $17.71$ 이 한계이고, $R=30$ 이면 $10.47$ 까지 내려가 목표를 지난다(§22 E7).

> 성격이 바뀐 것이 핵심이다. "어떤 설정으로도 도달 불가능"에서 **"ratio 를 올리면 도달하는 정상적인 조정 범위"** 로.

### 남은 과제

| 항목 | 상태 |
|---|---|
| $\sigma^{\text{ST}}$ 재도입(창 축소 / 분위수 곡선) | 보류 — CF 전환으로 필요성 낮아짐 (§22-B) |
| 학습/렌더 측정 지점 불일치(리버브) | **해결** — 손실을 리버브 뒤로 이동 |
| 학습/렌더 $\alpha,\rho$ 불일치 | 미해결 (§18 A). 리버브는 맞췄고 EQ·컴프 적용량은 그대로 |
| 레퍼런스 디리버브(`ref_dry`) — "웻+웻 대 웻" 비대칭 | 미착수 |
| 대역 경계 마스크 계단 | 부분 해결 (§20) |
| 소재별 적정 ratio 자동 추정 | 미착수 |
-->