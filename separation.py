"""
AI 음원 분리 모듈 — torchaudio 내장 Hybrid Demucs(HDEMUCS_HIGH_MUSDB_PLUS)를 사용해
레퍼런스 풀 곡에서 보컬 스템을 자동 추출한다.

별도 pip 패키지(demucs) 설치 불필요 — 이미 설치된 torchaudio에 모델이 포함돼 있으며,
가중치 파일만 첫 실행 시 자동 다운로드되어 torch hub 캐시에 저장된다.
CPU 추론이라 곡당 1~3분 소요될 수 있다(GPU 없음).
"""
import numpy as np
import torch
import librosa
import soundfile as sf
from torchaudio.transforms import Fade
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

_bundle = HDEMUCS_HIGH_MUSDB_PLUS
_TARGET_SR = _bundle.sample_rate  # 44100
_model = None


def _peak_normalize(x, peak_db=-1.0):
    """분리 스템의 피크를 peak_db(기본 -1dBFS)로 정규화 → 클리핑 방지 + 헤드룸 확보."""
    p = float(np.max(np.abs(x)))
    if p < 1e-8:
        return x
    return (x / p * (10.0 ** (peak_db / 20.0))).astype(np.float32)


def _get_model():
    """모델 지연 로드(가중치 캐시서 1회 로드). 최초 호출 시 자동 다운로드."""
    global _model
    if _model is None:
        _model = _bundle.get_model().eval()
    return _model


def _separate_sources(model, mix, segment=10.0, overlap=0.1, sample_rate=_TARGET_SR):
    """
    긴 오디오를 segment(초) 청크로 나눠 모델에 통과시키고 fade로 overlap-add.
    메모리 폭주와 경계 아티팩트를 동시에 방지.
    mix: [1, channels, length]  →  return [1, n_sources, channels, length]
    """
    device = mix.device
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start, end = 0, chunk_len
    overlap_frames = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def separate_vocals(song_path, vocals_out_path, inst_out_path=None):
    """
    레퍼런스 곡에서 보컬 스템 추출(선택적으로 반주 스템도).

    - song_path 를 44.1kHz 스테레오로 로드 → 모델 정규화 → 분리 → 역정규화
    - 'vocals' 채널을 vocals_out_path 에 저장
    - inst_out_path 지정 시 나머지 스템(drums+bass+other) 합을 반주로 저장

    Returns: dict(vocals_path, [inst_path], sample_rate, sources)
    """
    model = _get_model()

    y, _ = librosa.load(song_path, sr=_TARGET_SR, mono=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    else:
        y = y[:2]

    wav = torch.from_numpy(y)

    # 모델 입력 정규화(혼합 신호 통계 기준)
    ref = wav.mean(0)
    mean = float(ref.mean())
    std = float(ref.std()) + 1e-8
    wav = (wav - mean) / std

    sources = _separate_sources(model, wav.unsqueeze(0))[0]  # [n_src, ch, len]
    sources = sources * std + mean

    src = dict(zip(model.sources, sources))
    vocals = _peak_normalize(src["vocals"].cpu().numpy())
    sf.write(vocals_out_path, vocals.T, _TARGET_SR)

    result = {
        "vocals_path": vocals_out_path,
        "sample_rate": _TARGET_SR,
        "sources": list(model.sources),
    }
    if inst_out_path is not None:
        inst = _peak_normalize(sum(src[s] for s in model.sources if s != "vocals").cpu().numpy())
        sf.write(inst_out_path, inst.T, _TARGET_SR)
        result["inst_path"] = inst_out_path

    return result
