#!/bin/bash
# Script to run the DDSP Vocal Auto-EQ program

echo "============================================="
echo "   DDSP Vocal Auto-EQ 프로그램 실행 스크립트   "
echo "============================================="

# 스크립트가 어디서 호출되든 프로젝트 디렉터리 기준으로 동작하게 한다
cd "$(dirname "$0")" || exit 1

# 가상환경이 존재하는지 확인
if [ ! -x "venv/bin/python" ]; then
    echo "오류: venv 가상환경이 존재하지 않습니다."
    echo "먼저 가상환경을 구성하고 의존성을 설치해야 합니다."
    exit 1
fi

echo "서버를 실행하고 있습니다..."
echo "웹 브라우저를 열고 http://127.0.0.1:8000 에 접속해 주세요."
echo "서버를 종료하려면 Ctrl + C를 누르세요."
echo "---------------------------------------------"

# `source venv/bin/activate` 후 `python3` 를 부르지 않는다. activate 는 PATH 를
# 바꾸는 방식이라 셸이 이전 python3 경로를 캐시하고 있으면 시스템 파이썬이 그대로
# 잡힌다. 실제로 그 경로로 서버가 떠서 venv 에만 있는 패키지를 못 찾는 일이 있었다.
# venv 파이썬을 직접 부르면 PATH 와 무관하게 항상 venv 환경이 된다.
venv/bin/python main.py
