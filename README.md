# AutoMix

보컬과 레퍼런스 오디오를 분석해 보컬 톤과 다이내믹을 맞추고, 선택적으로 반주와 합쳐 마스터링하는 웹 애플리케이션입니다. 리버브는 자동 분석 없이 사용자가 RT60과 Wet을 직접 지정합니다.

## 실행

Python 3.10 환경에서 의존성을 설치하고 실행합니다.

```bash
pip install -r requirements.txt
python main.py
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. `run.sh`는 프로젝트 내 `venv`가 준비되어 있을 때 실행할 수 있습니다. Docker 실행은 `docker build -t automix .` 및 `docker run -p 8000:8000 automix`을 사용합니다.

## 주요 흐름

- 원본 보컬과 레퍼런스 보컬을 업로드하고 E2E DSP 파이프라인에서 EQ와 컴프레서를 조정합니다.
- 리버브를 켜면 RT60(0.1–4.0초)과 Wet(0–100%)을 수동 적용합니다. 레퍼런스 리버브 추론이나 리버브 학습은 실행하지 않습니다.
- 선택적으로 Demucs로 풀 레퍼런스에서 보컬을 분리하고, 반주와 믹스한 뒤 마스터링합니다.

## 구성

- `main.py`: FastAPI API와 실행 진입점
- `pipeline.py`, `modules.py`, `losses.py`: 보컬 처리, 믹싱 및 마스터링 DSP
- `separation.py`: 선택적 레퍼런스 보컬 분리
- `static/`: 웹 UI
