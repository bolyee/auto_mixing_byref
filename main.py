import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from modules import mix_with_instrumental, master_track
from pipeline import match_e2e, VOCAL_EQ_BANDS
from separation import separate_vocals

app = FastAPI(title="DDSP Vocal Auto Equalizer")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount static directories
# Files in data/outputs/ will be served at /data/outputs/
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.post("/api/match-eq")
async def api_match_eq(
    raw_vocal: UploadFile = File(...),
    ref_vocal: UploadFile = File(...),
    # EQ 밴드 수. 밴드는 200 Hz~10 kHz 로그 등간격이다(그 밖은 L_tone 평가 대역 Ω 밖).
    # 기본값은 `pipeline.VOCAL_EQ_BANDS` 와 같아야 한다 — UI 슬라이더 기본도 17 이다.
    num_bands: int = Form(VOCAL_EQ_BANDS),
    # EQ 정규화(페널티) 계수 — 밴드 게인의 크기/거칠기를 억제한다.
    #   eq_l2:     mean(g^2) 페널티. 크면 전체 부스트/컷 폭이 작아진다.
    #   eq_smooth: mean(diff(g)^2) 페널티. 크면 인접 밴드 간 요철이 줄어 커브가 매끈해진다.
    eq_l2: float = Form(0.01),
    eq_smooth: float = Form(0.1),
    # 컴프 ratio. 학습하지 않는 설정값(1=무압축, 3=보컬 표준, 큰 값=리미터).
    comp_ratio: float = Form(3.0),
    # 컴프 threshold 가 신호 아래로 내려가는 것을 막는 정규화 가중치. 0이면 비활성.
    comp_thresh_weight: float = Form(0.0),
    match_amount: float = Form(1.0),
    match_volume: bool = Form(False),
    comp_amount: float = Form(1.0),
    reverb_amount: float = Form(1.0),
    mode: str = Form("both"),
    instrumental: UploadFile = File(None),
    vocal_gain_db: float = Form(0.0),
    separate_ref: bool = Form(False),
    master: bool = Form(False),
    e2e_steps: int = Form(50),
    # 잔향(IR)을 어디서 측정할지: "instrumental"(반주, 기본) 또는 "reference"(레퍼런스 보컬)
    ir_source: str = Form("instrumental"),
    # 잔향 생성 방식
    #   "measured" — Rec-RIR 이 추정한 IR 을 그대로 컨볼브
    #   "synth"    — 레퍼런스에서 rt60 추출 후 wet 을 학습
    #   "manual"   — 아래 manual_rt60 / manual_wet 을 그대로 사용 (추정·학습 없음) **기본**
    # 기본이 manual 인 이유: IR 추정이 최초 약 60초 걸리고, 새 세션에서 바로 결과를 보려면
    # 추정 단계가 없는 쪽이 낫다. UI 토글 기본값(`static/index.html` rmode-manual)과 맞춘다.
    reverb_mode: str = Form("manual"),
    # 수동 모드 전용. RT60 은 [0.1, 4.0]s, wet 은 [0, 1] 로 클립된다.
    manual_rt60: float = Form(1.0),
    manual_wet: float = Form(0.3),
    # True 면 리버브 4조합(measured/synth × instrumental/reference)을 모두 렌더해
    # 웹에서 들어보며 비교할 수 있게 한다.
    compare_reverbs: bool = Form(False)
):
    # Validate file extensions (instrumental optional)
    files_to_check = [raw_vocal, ref_vocal]
    if instrumental is not None and instrumental.filename:
        files_to_check.append(instrumental)
    for audio_file in files_to_check:
        ext = os.path.splitext(audio_file.filename)[1].lower()
        if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}. Please upload WAV, MP3, FLAC, OGG, or M4A.")

    # Generate unique filenames to avoid collision
    session_id = str(uuid.uuid4())
    raw_filename = f"raw_{session_id}_{raw_vocal.filename}"
    ref_filename = f"ref_{session_id}_{ref_vocal.filename}"
    out_filename = f"output_{session_id}.wav"

    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    ref_path = os.path.join(UPLOADS_DIR, ref_filename)
    out_path = os.path.join(OUTPUTS_DIR, out_filename)

    has_instrumental = instrumental is not None and bool(instrumental.filename)
    inst_path = None
    fullmix_path = None
    fullmix_raw_path = None
    mastered_path = None
    ref_extracted_path = None
    if has_instrumental:
        inst_filename = f"inst_{session_id}_{instrumental.filename}"
        inst_path = os.path.join(UPLOADS_DIR, inst_filename)
        fullmix_path = os.path.join(OUTPUTS_DIR, f"fullmix_{session_id}.wav")
        fullmix_raw_path = os.path.join(OUTPUTS_DIR, f"fullmixraw_{session_id}.wav")
        if master:
            mastered_path = os.path.join(OUTPUTS_DIR, f"mastered_{session_id}.wav")
    
    try:
        # Save uploaded files
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(raw_vocal.file, buffer)
            
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(ref_vocal.file, buffer)

        if has_instrumental:
            with open(inst_path, "wb") as buffer:
                shutil.copyfileobj(instrumental.file, buffer)

        # Optional: extract vocal from a full reference song via Demucs (torchaudio)
        # → use the extracted vocal stem as the matching reference.
        ref_for_match = ref_path
        ref_extracted_path = None
        if separate_ref:
            ref_extracted_path = os.path.join(OUTPUTS_DIR, f"refvocals_{session_id}.wav")
            separate_vocals(ref_path, ref_extracted_path)
            ref_for_match = ref_extracted_path

        # Run EQ Matching
        def _run(out, r_mode, i_src):
            return match_e2e(
                raw_path=raw_path,
                ref_path=ref_for_match,
                out_path=out,
                num_bands=num_bands,
                eq_l2=eq_l2,
                eq_smooth=eq_smooth,
                comp_ratio=comp_ratio,
                comp_thresh_weight=comp_thresh_weight,
                match_amount=match_amount,
                match_volume=match_volume,
                max_gain_db=15.0,
                comp_amount=comp_amount,
                reverb_amount=reverb_amount,
                mode=mode,
                n_steps=e2e_steps,
                ir_source=i_src,
                reverb_mode=r_mode,
                manual_rt60=manual_rt60,
                manual_wet=manual_wet,
                inst_path=inst_path,
            )

        reverb_variants = {}
        # 4조합 비교는 자동 모드(measured/synth × 반주/레퍼런스)의 비교다. 수동 모드는
        # 추정 대상이 없어 비교할 축이 없으므로, 켜져 있어도 무시하고 지정값으로 1회만 돌린다.
        if compare_reverbs and reverb_mode == "manual":
            compare_reverbs = False
        if compare_reverbs:
            # 리버브 4조합을 각각 돌린다. 첫 조합의 결과를 대표(result)로 쓰고
            # 나머지는 비교용 오디오로만 남긴다.
            combos = [
                ("measured", "instrumental"),
                ("measured", "reference"),
                ("synth", "instrumental"),
                ("synth", "reference"),
            ]
            result = None
            for r_mode, i_src in combos:
                key = f"{r_mode}_{i_src}"
                vpath = os.path.join(OUTPUTS_DIR, f"rv_{key}_{session_id}.wav")
                try:
                    res = _run(vpath, r_mode, i_src)
                except Exception as exc:
                    print(f"[compare] {key} 실패: {exc}")
                    continue
                rv = res.get("reverb_data", {}) or {}
                reverb_variants[key] = {
                    "url": f"/data/outputs/rv_{key}_{session_id}.wav",
                    "reverb_mode": r_mode,
                    "ir_source": i_src,
                    "rt60": rv.get("rt60", 0.0),
                    "wet": rv.get("wet", 0.0),
                    "n_segments": rv.get("n_segments", 0),
                    "active": rv.get("active", False),
                }
                if result is None:
                    result = res
                    shutil.copyfile(vpath, out_path)
            if result is None:
                raise RuntimeError("리버브 비교 조합이 모두 실패했습니다.")
        else:
            result = _run(out_path, reverb_mode, ir_source)
        
        # Mix with instrumental → full mix (optional). 처리후/처리전 두 버전 생성(발표용 A/B).
        fullmix_info = None
        if has_instrumental:
            fullmix_info = mix_with_instrumental(
                vocal_path=out_path,
                inst_path=inst_path,
                out_path=fullmix_path,
                vocal_gain_db=vocal_gain_db
            )
            # 처리 전(원본 raw 보컬) + 반주 — 동일 RMS 밸런스로 레벨 맞춰 공정 비교
            mix_with_instrumental(
                vocal_path=raw_path,
                inst_path=inst_path,
                out_path=fullmix_raw_path,
                vocal_gain_db=vocal_gain_db
            )

        # Mastering (optional): 풀믹스 톤/라우드니스를 레퍼런스 곡에 맞추고 브릭월 리미터
        mastering_info = None
        if has_instrumental and master:
            mastering_info = master_track(
                mix_path=fullmix_path,
                ref_path=ref_path,       # 업로드한 레퍼런스 곡(풀 곡) 기준
                out_path=mastered_path
            )

        # Prepare response
        raw_url = f"/data/uploads/{raw_filename}"
        ref_url = f"/data/uploads/{ref_filename}"
        output_url = f"/data/outputs/{out_filename}"

        # 레퍼런스 분리 시: REFERENCE 는 추출된 보컬(매칭 대상)을 재생, 원곡은 ref_song 로 별도 제공
        if separate_ref and ref_extracted_path:
            ref_url = f"/data/outputs/refvocals_{session_id}.wav"

        audio_urls = {
            "raw": raw_url,
            "ref": ref_url,
            "processed": output_url
        }
        if separate_ref and ref_extracted_path:
            audio_urls["ref_song"] = f"/data/uploads/{ref_filename}"
        if has_instrumental:
            audio_urls["instrumental"] = f"/data/uploads/{inst_filename}"
            audio_urls["full_mix"] = f"/data/outputs/fullmix_{session_id}.wav"
            audio_urls["full_mix_raw"] = f"/data/outputs/fullmixraw_{session_id}.wav"
            if master and mastered_path:
                audio_urls["full_mix_mastered"] = f"/data/outputs/mastered_{session_id}.wav"

        return {
            "status": "success",
            "engine": "e2e",
            # 어떤 페널티로 돌렸는지 결과와 같이 돌려줘 A/B 비교 때 기록으로 쓴다
            "penalties": {
                "eq_l2": eq_l2,
                "eq_smooth": eq_smooth,
                "comp_ratio": comp_ratio,
                "comp_thresh_weight": comp_thresh_weight,
            },
            "reverb_variants": reverb_variants,
            "audio_urls": audio_urls,
            "separated_ref": bool(separate_ref and ref_extracted_path),
            "match_error": result["match_error"],
            "tonal_error": result["tonal_error"],
            "dynamics_error": result["dynamics_error"],
            "reverb_error": result.get("reverb_error", 0.0),
            "chart_data": result["chart_data"],
            "compression_data": result["compression_data"],
            "reverb_data": result.get("reverb_data"),
            "gate_data": result.get("gate_data"),
            "fullmix_data": fullmix_info,
            "mastering_data": mastering_info
        }
        
    except Exception as e:
        # Clean up files if error occurs
        for p in [raw_path, ref_path, out_path, inst_path, fullmix_path, fullmix_raw_path, mastered_path, ref_extracted_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        raise HTTPException(status_code=500, detail=f"EQ Matching failed: {str(e)}")

# Serve the static web interface
@app.get("/")
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)

# Serve general static files (JS, CSS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
