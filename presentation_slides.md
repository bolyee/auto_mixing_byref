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

# AutoMix 기술 설명

레퍼런스 추종 End-to-End 믹싱 & 마스터링 시스템

이현중 · 2026.10 · 복사본 2 현재 구현

---

## 1. 처리 범위

- 입력: 원본 보컬, 레퍼런스, 선택 반주
- 자동: EQ 밴드 게인, 컴프 threshold, 마스터링 EQ·드라이브
- 수동: 리버브 RT60·Wet, 컴프 ratio, 매칭 강도, 보컬 게인
- 선택: Demucs 보컬 분리, 반주 합산, 마스터링
- 별도의 보컬 매칭 신경망을 학습하지 않고 DSP 파라미터를 입력별 최적화

---

## 2. 신호 흐름

```text
44.1 kHz 모노 / 피크 정규화
 → STFT → EQ → iSTFT → 컴프 → 수동 리버브(선택)
 → 소프트 리미터 → 보컬 WAV·지표
 → 반주 합산(선택) → 마스터링(선택) → 최종 WAV
```

분리한 레퍼런스 보컬은 보컬 처리의 목표이다.
마스터링은 업로드 원본 레퍼런스를 목표로 사용한다.
반주가 없으면 처리 보컬 자체를 마스터링한다.

---

## 3. 보컬 EQ와 톤 손실

- 17밴드, 200 Hz–10 kHz, STFT 2,048 / hop 512 / Hann
- 밴드 dB 게인은 $g=15\theta$, 가우시안 필터뱅크 합으로 커브 구성
- 80개 멜 필터의 전체 시간평균 파워를 dB로 변환
- 평가 대역 평균 dB를 빼서 음량 차이를 정렬한 뒤 L1 오차
- EQ 정규화: $0.1\operatorname{mean}(g^2)+0.1\operatorname{mean}((\Delta g)^2)$
- 보컬 EQ에는 별도 에너지 중립화 단계가 없음

---

## 4. 컴프레서와 CF

- 시간영역 피크 검출 → soft knee → 목표 GR → attack/release 평활화
- threshold 범위 −60~0 dB, 초기 −30 dB; ratio 기본 10:1 설정값
- attack 0.5 ms / release 120 ms / knee 4 dB
- 활성 구간 평균 GR을 보상하는 오토 메이크업
- CF = 4배 오버샘플 True Peak(dB) − RMS(dB)
- 손실: $|CF(y)-CF(r)|$, 각 음원의 최대 LUFS 10초로 평가
- True Peak 순전파는 max, 경사는 상위 표본 평균으로 분산

---

## 5. 경사 분리와 보컬 50회

- 톤 경로: 컴프 파라미터 detach → EQ에만 역전파
- CF 경로: EQ 출력 detach → 컴프 threshold에만 역전파
- 활성 리버브는 두 경로의 최종 신호에 반영
- 두 손실을 각각 backward 후 단일 Adam 파라미터 그룹 갱신
- EQ·컴프 학습률 각각 0.05, 기본 50회
- CF는 10초에서 측정하지만 EQ·컴프 순전파는 전체 음원에 적용

---

## 6. 수동 리버브

- 독립된 좌우 잡음에 지수 감쇠를 적용한 합성 IR
- RT60 0.1–4.0초, Wet 0–1; API 기본 1.0초·0.2
- 파라미터 고정, 리버브 자동 추론·디리버브 없음
- IR 에너지 정규화 후 FFT 컨볼루션, dry/wet 혼합
- IR 길이 1.5초, 체인 출력은 입력 길이로 잘림
- 감쇠 오차는 출력 분석 지표이며 리버브 학습에 사용하지 않음

---

## 7. 화면·내부 평가 통일

- 보컬 학습과 차트는 같은 torch STFT의 reflect 패딩 사용
- 같은 Slaney 멜 필터·평균 dB 정렬 함수 공유
- 실제 필터 무게중심으로 200 Hz 이상 10 kHz 미만 선택
- 내부 최종 평가와 화면 모두 소프트 리미터 이후 신호 사용
- 화면 tonal_error = 내부 최종 L_tone_raw
- match_error = 0.65 × tonal_error + 0.35 × dynamics_error
- 이 합산 표시값은 학습 목적함수의 가중치가 아님

---

## 8. 반주 합산

- t=0 정렬을 가정, 짧은 쪽 끝을 0 패딩
- 반주 피크 −6 dBFS 정규화
- 보컬·반주의 최대 LUFS 3초를 비교하고 사용자 게인 반영
- API 보컬 게인 기본 −2 dB; 함수 단독 기본값은 0 dB
- 합산 출력에 −1 dBFS 헤드룸 기준 소프트 제한
- 자동 박자 정렬·보컬 위치 탐색은 수행하지 않음

---

## 9. 통합 마스터링: 같은 루프에서 300회

| 대상 | 구성 | 학습률 | 반복 |
| --- | --- | --- | --- |
| EQ | 15밴드, 20 Hz–10 kHz, 정규화 0.5 / 0.5 | 0.05 | 300회 |
| 드라이브 | 리미터 입력 게인(dB) | 0.1 | 300회 |

- 톤: 전체 곡 평균 파워 기반 L1 + EQ 정규화
- EQ 전체 에너지 변화 중립화 후 선택 10초를 렌더링
- EQ·드라이브 각각 Adam, 같은 루프에서 한 번 합산 backward
- LUFS·감쇄 경로의 EQ 출력은 detach

---

## 10. 드라이브 목적과 감쇄 가드

$$L_{drive}=(LUFS(y_{lim})-LUFS(r))^2+25[\max(0,GR_{max}-3)]^2$$

- LUFS는 미분 가능한 채널 에너지 합산·소프트 게이팅 미터
- 리미터: ratio 50:1 / attack 0.1 ms / release 80 ms / knee 1 dB
- 샘플 실링 −0.3 dBFS, True Peak 적합성은 별도 검증
- Adam 이후 선택 10초에서 3.5 dB 기준 추가 가드
- 최대 80회, 0.25 dB씩 드라이브 감쇠, 하한 −12 dB
- 전곡 GR 상한이 아니며 목표 LUFS 도달도 보장하지 않음

---

## 11. 최신 평가 일치 검사

3초 합성 신호, 보컬 50회; 실제 보컬 성능 평가와 구분

| 조건 | 내부 음색 오차 | 화면 음색 오차 |
| --- | ---: | ---: |
| EQ+컴프 | 2.692196 | 2.692196 |
| 리버브 포함 | 3.085835 | 3.085835 |
| 리미터 강제 작동 | 5.620083 | 5.620083 |

차트 재계산값도 1e-5 이내 일치.
리미터 강제 작동은 입력을 10배 키운 회귀 검사이다.
재현: 상위 `최종점검/check_tone_consistency.py`.

---

## 12. 검증 범위와 남은 자료

- 합성 신호에서 보컬·리버브·반주 합산·300회 마스터링 실행 확인
- 현재 코드의 실제 음원·브라우저·Demucs 시연은 별도 확인
- 과거 실제 보컬·블라인드 테스트 기록은 당시 버전·조건과 함께 사용
- 전역 최적해, 모든 입력의 음질 개선, 플랫폼 규격 준수를 보장하지 않음
- 이전 실험 슬라이드는 상위 문서_이전버전_2026-10-21 폴더에 보관
