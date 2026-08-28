import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ddsp import match_eq, mix_with_instrumental, master_track
from pipeline import match_e2e
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
    num_bands: int = Form(5),
    match_amount: float = Form(1.0),
    smoothness: float = Form(1.0),
    hpf_freq: float = Form(80.0),
    lpf_freq: float = Form(16000.0),
    match_volume: bool = Form(False),
    comp_amount: float = Form(1.0),
    reverb_amount: float = Form(1.0),
    mode: str = Form("both"),
    instrumental: UploadFile = File(None),
    vocal_gain_db: float = Form(0.0),
    separate_ref: bool = Form(False),
    master: bool = Form(False),
    # "e2e"  = EQ/Comp/Reverb 를 단일 그래프에서 통합 손실로 동시 최적화 (pipeline.match_e2e)
    # "legacy" = 기존 순차 최적화 (ddsp.match_eq)
    engine: str = Form("e2e"),
    e2e_steps: int = Form(100)
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
        # Convert front-end HPF/LPF values of 0 or high limits to None
        hpf = None if hpf_freq <= 20 else hpf_freq
        lpf = None if lpf_freq >= 20000 else lpf_freq

        engine_fn = match_e2e if engine == "e2e" else match_eq
        engine_kwargs = dict(
            raw_path=raw_path,
            ref_path=ref_for_match,
            out_path=out_path,
            num_bands=num_bands,
            match_amount=match_amount,
            smoothness=smoothness,
            hpf_freq=hpf,
            lpf_freq=lpf,
            match_volume=match_volume,
            max_gain_db=15.0,
            comp_amount=comp_amount,
            reverb_amount=reverb_amount,
            mode=mode
        )
        if engine == "e2e":
            engine_kwargs["n_steps"] = e2e_steps

        result = engine_fn(**engine_kwargs)
        
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
            "engine": engine,
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
