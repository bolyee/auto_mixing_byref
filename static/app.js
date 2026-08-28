// Frontend Application Logic for DDSP Vocal Auto-EQ

// State Variables
let rawFile = null;
let refFile = null;
let instFile = null;
let activeMode = 'raw'; // 'raw', 'processed', 'ref'
let currentAudio = null;
let chartInstance = null;
let masterChartInstance = null;
let isSeeking = false;

// Audio Elements
const audioRaw = document.getElementById('audio-raw');
const audioProcessed = document.getElementById('audio-processed');
const audioRef = document.getElementById('audio-ref');
const audioFullmix = document.getElementById('audio-fullmix');
const audioFullmixRaw = document.getElementById('audio-fullmix-raw');
const audioMastered = document.getElementById('audio-mastered');

// DOM Elements
const rawUploadBox = document.getElementById('raw-upload-box');
const refUploadBox = document.getElementById('ref-upload-box');
const rawFileInput = document.getElementById('raw-file-input');
const refFileInput = document.getElementById('ref-file-input');
const rawFileName = document.getElementById('raw-file-name');
const refFileName = document.getElementById('ref-file-name');
const instUploadBox = document.getElementById('inst-upload-box');
const instFileInput = document.getElementById('inst-file-input');
const instFileName = document.getElementById('inst-file-name');

const matchAmountSlider = document.getElementById('match-amount');
const matchAmountVal = document.getElementById('match-amount-val');
// 잔향은 매칭 강도와 분리한다. 추정 IR 의 DRR 이 매우 젖게 나오는 경우가 있어
// 리버브만 줄여야 하는데, 하나로 묶으면 EQ·컴프까지 같이 약해진다.
const reverbAmountSlider = document.getElementById('reverb-amount');
const reverbAmountVal = document.getElementById('reverb-amount-val');
const eqBandsSlider = document.getElementById('eq-bands');
const eqBandsVal = document.getElementById('eq-bands-val');
// 손실 정규화(페널티) 계수. 개발 중 값을 자주 바꿔가며 튜닝하므로 UI 로 뺐다.
const eqL2Slider = document.getElementById('eq-l2');
const eqL2Val = document.getElementById('eq-l2-val');
const eqSmoothSlider = document.getElementById('eq-smooth');
const eqSmoothVal = document.getElementById('eq-smooth-val');
const compRatioSlider = document.getElementById('comp-ratio');
const compRatioVal = document.getElementById('comp-ratio-val');
const compThreshSlider = document.getElementById('comp-thresh');
const compThreshVal = document.getElementById('comp-thresh-val');
const vocalGainSlider = document.getElementById('vocal-gain');
const vocalGainVal = document.getElementById('vocal-gain-val');

const processBtn = document.getElementById('process-btn');
const btnLoader = document.getElementById('btn-loader');
const downloadBtn = document.getElementById('download-btn');

const masterPlayBtn = document.getElementById('master-play-btn');
const globalSeek = document.getElementById('global-seek');
const currentTimeDisplay = document.getElementById('current-time');
const durationDisplay = document.getElementById('duration');
const masterVolume = document.getElementById('master-volume');

const modeBtnRaw = document.getElementById('mode-raw');
const modeBtnProcessed = document.getElementById('mode-processed');
const modeBtnRef = document.getElementById('mode-ref');
const modeBtnFullmix = document.getElementById('mode-fullmix');
const modeBtnFullmixRaw = document.getElementById('mode-fullmix-raw');
const modeBtnMastered = document.getElementById('mode-mastered');

// 1. Parameter Display Sync
matchAmountSlider.addEventListener('input', (e) => {
    matchAmountVal.textContent = Math.round(e.target.value * 100) + '%';
});
reverbAmountSlider.addEventListener('input', (e) => {
    reverbAmountVal.textContent = Math.round(e.target.value * 100) + '%';
});
eqBandsSlider.addEventListener('input', (e) => {
    eqBandsVal.textContent = e.target.value;
});
eqL2Slider.addEventListener('input', (e) => {
    eqL2Val.textContent = parseFloat(e.target.value).toFixed(4);
});
eqSmoothSlider.addEventListener('input', (e) => {
    eqSmoothVal.textContent = parseFloat(e.target.value).toFixed(4);
});
if (compRatioSlider) compRatioSlider.addEventListener('input', (e) => {
    compRatioVal.textContent = `${parseFloat(e.target.value).toFixed(1)} : 1`;
});
compThreshSlider.addEventListener('input', (e) => {
    compThreshVal.textContent = parseFloat(e.target.value).toFixed(4);
});
vocalGainSlider.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    vocalGainVal.textContent = `${v > 0 ? '+' : ''}${v.toFixed(1)} dB`;
});

// 처리 모듈 토글 세 개. `.toggle-btn` 클래스는 리버브 방식(rmode-*)과 잔향 측정
// 대상(rsrc-*) 버튼도 함께 쓰므로, 클래스로 긁으면 그 버튼들까지 딸려 온다.
const MODULE_TOGGLE_IDS = ['toggle-eq', 'toggle-comp', 'toggle-reverb'];

function activeModuleButtons() {
    return MODULE_TOGGLE_IDS
        .map(id => document.getElementById(id))
        .filter(el => el && el.classList.contains('active'));
}

// Toggle button click logic
// rmode-* / rsrc-* 는 여기서 다루지 않는다. 그쪽은 bindExclusiveToggle 이 라디오처럼
// 처리하므로 이 핸들러가 끼어들면 토글이 두 번 일어난다.
MODULE_TOGGLE_IDS
    .map(id => document.getElementById(id))
    .filter(Boolean)
    .forEach(btn => {
        btn.addEventListener('click', () => {
            btn.classList.toggle('active');

            // Ensure at least one module is active to prevent processing nothing
            if (activeModuleButtons().length === 0) {
                btn.classList.add('active');
                alert('최소 하나의 처리 모듈(EQ, 컴프레서, 리버브 중 하나)은 활성화되어야 합니다.');
            }
        });
    });

// 2. Upload Box Handlers
function setupUploadBox(box, input, nameDisplay, onFileSelect) {
    box.addEventListener('click', () => input.click());
    
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            onFileSelect(file);
            nameDisplay.textContent = file.name;
            box.classList.add('has-file');
            checkReadyToProcess();
        }
    });

    // Drag and Drop
    box.addEventListener('dragover', (e) => {
        e.preventDefault();
        box.classList.add('dragover');
    });

    box.addEventListener('dragleave', () => {
        box.classList.remove('dragover');
    });

    box.addEventListener('drop', (e) => {
        e.preventDefault();
        box.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('audio/') || file.name.match(/\.(wav|mp3|flac|ogg|m4a)$/i)) {
                input.files = e.dataTransfer.files;
                onFileSelect(file);
                nameDisplay.textContent = file.name;
                box.classList.add('has-file');
                checkReadyToProcess();
            } else {
                alert('지원되는 오디오 파일 형식만 업로드 가능합니다.');
            }
        }
    });
}

setupUploadBox(rawUploadBox, rawFileInput, rawFileName, (file) => rawFile = file);
setupUploadBox(refUploadBox, refFileInput, refFileName, (file) => refFile = file);
setupUploadBox(instUploadBox, instFileInput, instFileName, (file) => instFile = file);

function checkReadyToProcess() {
    if (rawFile && refFile) {
        processBtn.removeAttribute('disabled');
    } else {
        processBtn.setAttribute('disabled', 'true');
    }
}

// 3. Audio Player Logic (A/B Switcher)
function formatTime(seconds) {
    if (isNaN(seconds)) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function stopAllAudio() {
    audioRaw.pause();
    audioProcessed.pause();
    audioRef.pause();
    audioFullmix.pause();
    audioFullmixRaw.pause();
    audioMastered.pause();
}

function syncPlayState() {
    if (!currentAudio) return;
    
    if (currentAudio.paused) {
        masterPlayBtn.querySelector('.play-icon').textContent = '▶';
    } else {
        masterPlayBtn.querySelector('.play-icon').textContent = '⏸';
    }
}

function switchMode(newMode) {
    if (!currentAudio) return;
    
    const wasPlaying = !currentAudio.paused;
    const currentTime = currentAudio.currentTime;
    
    stopAllAudio();
    
    activeMode = newMode;

    // Select new active audio + sync button highlight (4 modes)
    const audioMap = { raw: audioRaw, processed: audioProcessed, ref: audioRef, fullmix: audioFullmix, fullmixraw: audioFullmixRaw, mastered: audioMastered };
    const btnMap = { raw: modeBtnRaw, processed: modeBtnProcessed, ref: modeBtnRef, fullmix: modeBtnFullmix, fullmixraw: modeBtnFullmixRaw, mastered: modeBtnMastered };
    currentAudio = audioMap[activeMode] || audioProcessed;
    Object.keys(btnMap).forEach((m) => btnMap[m].classList.toggle('active', m === activeMode));

    // Synchronize play parameters
    currentAudio.currentTime = currentTime;
    currentAudio.volume = masterVolume.value;
    
    if (wasPlaying) {
        currentAudio.play().catch(err => console.log("Auto-play error:", err));
    }
    
    syncPlayState();
    updateDurationDisplay();
}

function updateDurationDisplay() {
    if (currentAudio && !isNaN(currentAudio.duration)) {
        durationDisplay.textContent = formatTime(currentAudio.duration);
        globalSeek.max = Math.floor(currentAudio.duration);
    }
}

// Master Controls Bindings
masterPlayBtn.addEventListener('click', () => {
    if (!currentAudio) return;
    
    if (currentAudio.paused) {
        currentAudio.play().catch(err => console.log("Play error:", err));
    } else {
        currentAudio.pause();
    }
    syncPlayState();
});

// Update Seek Bar & Current Time label
function handleTimeUpdate(e) {
    if (e.target !== currentAudio || isSeeking) return;
    
    globalSeek.value = Math.floor(currentAudio.currentTime);
    currentTimeDisplay.textContent = formatTime(currentAudio.currentTime);
}

audioRaw.addEventListener('timeupdate', handleTimeUpdate);
audioProcessed.addEventListener('timeupdate', handleTimeUpdate);
audioRef.addEventListener('timeupdate', handleTimeUpdate);
audioFullmix.addEventListener('timeupdate', handleTimeUpdate);
audioFullmixRaw.addEventListener('timeupdate', handleTimeUpdate);
audioMastered.addEventListener('timeupdate', handleTimeUpdate);

// Sync duration on load metadata
function handleMetadataLoad(e) {
    if (e.target === currentAudio) {
        updateDurationDisplay();
    }
}

audioRaw.addEventListener('loadedmetadata', handleMetadataLoad);
audioProcessed.addEventListener('loadedmetadata', handleMetadataLoad);
audioRef.addEventListener('loadedmetadata', handleMetadataLoad);
audioFullmix.addEventListener('loadedmetadata', handleMetadataLoad);
audioFullmixRaw.addEventListener('loadedmetadata', handleMetadataLoad);
audioMastered.addEventListener('loadedmetadata', handleMetadataLoad);

// Audio ends -> pause icon
function handleAudioEnded(e) {
    if (e.target === currentAudio) {
        syncPlayState();
    }
}

audioRaw.addEventListener('ended', handleAudioEnded);
audioProcessed.addEventListener('ended', handleAudioEnded);
audioRef.addEventListener('ended', handleAudioEnded);
audioFullmix.addEventListener('ended', handleAudioEnded);
audioFullmixRaw.addEventListener('ended', handleAudioEnded);
audioMastered.addEventListener('ended', handleAudioEnded);

// Seek Bar Dragging
globalSeek.addEventListener('input', () => {
    isSeeking = true;
    currentTimeDisplay.textContent = formatTime(globalSeek.value);
});

globalSeek.addEventListener('change', () => {
    const seekVal = parseFloat(globalSeek.value);
    
    // Seek ALL audios so they remain perfectly in sync
    audioRaw.currentTime = seekVal;
    audioProcessed.currentTime = seekVal;
    audioRef.currentTime = seekVal;
    audioFullmix.currentTime = seekVal;
    audioFullmixRaw.currentTime = seekVal;
    audioMastered.currentTime = seekVal;

    isSeeking = false;
});

// Volume Adjustment
masterVolume.addEventListener('input', (e) => {
    const vol = e.target.value;
    audioRaw.volume = vol;
    audioProcessed.volume = vol;
    audioRef.volume = vol;
    audioFullmix.volume = vol;
    audioFullmixRaw.volume = vol;
    audioMastered.volume = vol;
});

// Mode buttons click listener
modeBtnRaw.addEventListener('click', () => switchMode('raw'));
modeBtnProcessed.addEventListener('click', () => switchMode('processed'));
modeBtnRef.addEventListener('click', () => switchMode('ref'));
modeBtnFullmix.addEventListener('click', () => switchMode('fullmix'));
modeBtnFullmixRaw.addEventListener('click', () => switchMode('fullmixraw'));
modeBtnMastered.addEventListener('click', () => switchMode('mastered'));

// 4. API Request & Equalization Processing
processBtn.addEventListener('click', async () => {
    if (!rawFile || !refFile) return;
    
    // Loading State UI
    processBtn.setAttribute('disabled', 'true');
    btnLoader.removeAttribute('hidden');
    processBtn.querySelector('.btn-text').textContent =
        document.getElementById('separate-ref-check').checked ? 'AI 보컬 분리 + 최적화 중... (1~3분)' : '스펙트럼 최적화 분석 중...';
    
    const formData = new FormData();
    formData.append('raw_vocal', rawFile);
    formData.append('ref_vocal', refFile);
    formData.append('num_bands', eqBandsSlider.value);
    formData.append('eq_l2', eqL2Slider.value);
    formData.append('eq_smooth', eqSmoothSlider.value);
    if (compRatioSlider) formData.append('comp_ratio', compRatioSlider.value);
    formData.append('comp_thresh_weight', compThreshSlider.value);
    formData.append('match_amount', matchAmountSlider.value);
    formData.append('smoothness', '1.0');
    formData.append('match_volume', 'true');
    formData.append('comp_amount', matchAmountSlider.value);
    formData.append('reverb_amount', reverbAmountSlider.value);
    formData.append('vocal_gain_db', vocalGainSlider.value);
    if (instFile) {
        formData.append('instrumental', instFile);
    }
    const separateRef = document.getElementById('separate-ref-check').checked;
    formData.append('separate_ref', separateRef ? 'true' : 'false');
    formData.append('master', document.getElementById('master-check').checked ? 'true' : 'false');

    // Construct mode string from active module toggles.
    // `.toggle-btn.active` 로 긁으면 안 된다. 리버브 방식(rmode-*)과 잔향 측정
    // 대상(rsrc-*) 버튼도 같은 클래스를 쓰기 때문에 그 값까지 모드 문자열에 섞여
    // 서버에 `eq,comp,reverb,measured,instrumental` 로 전달된다.
    const activeModes = activeModuleButtons().map(el => el.dataset.value);
    const mixMode = activeModes.join(',');
    formData.append('mode', mixMode);
    // 결과 카드에서 다시 참조하므로 변수로 잡아 둔다 (응답에는 담겨오지 않는다)
    const reverbMethod = selectedValue(['rmode-measured', 'rmode-synth', 'rmode-manual'], 'manual');
    const irSource = selectedValue(['rsrc-instrumental', 'rsrc-reference'], 'instrumental');
    formData.append('reverb_mode', reverbMethod);
    formData.append('ir_source', irSource);
    // 수동 모드 값. 다른 모드에서는 서버가 무시한다.
    if (manualRt60Slider) formData.append('manual_rt60', manualRt60Slider.value);
    if (manualWetSlider) formData.append('manual_wet', manualWetSlider.value);
    const cmpEl = document.getElementById('compare-reverbs-check');
    const wantCompare = reverbMethod !== 'manual' && cmpEl && cmpEl.checked;
    formData.append('compare_reverbs', wantCompare ? 'true' : 'false');
    
    try {
        const response = await fetch('/api/match-eq', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || '이퀄라이징 처리에 실패했습니다.');
        }
        
        const data = await response.json();
        
        // Setup audio elements sources
        audioRaw.src = data.audio_urls.raw;
        audioProcessed.src = data.audio_urls.processed;
        audioRef.src = data.audio_urls.ref;

        // Load the audio
        audioRaw.load();
        audioProcessed.load();
        audioRef.load();

        // Full mix (반주 합침) — 있을 때만 활성화 (처리후 + 처리전 두 버전 + 마스터)
        const hasFullMix = !!(data.audio_urls && data.audio_urls.full_mix);
        const hasMastered = !!(data.audio_urls && data.audio_urls.full_mix_mastered);
        if (hasFullMix) {
            audioFullmix.src = data.audio_urls.full_mix;
            audioFullmix.load();
            modeBtnFullmix.style.display = '';
            modeBtnFullmix.removeAttribute('disabled');
            if (data.audio_urls.full_mix_raw) {
                audioFullmixRaw.src = data.audio_urls.full_mix_raw;
                audioFullmixRaw.load();
                modeBtnFullmixRaw.style.display = '';
                modeBtnFullmixRaw.removeAttribute('disabled');
            }
        } else {
            modeBtnFullmix.style.display = 'none';
            modeBtnFullmix.setAttribute('disabled', 'true');
            modeBtnFullmixRaw.style.display = 'none';
            modeBtnFullmixRaw.setAttribute('disabled', 'true');
        }
        if (hasMastered) {
            audioMastered.src = data.audio_urls.full_mix_mastered;
            audioMastered.load();
            modeBtnMastered.style.display = '';
            modeBtnMastered.removeAttribute('disabled');
        } else {
            modeBtnMastered.style.display = 'none';
            modeBtnMastered.setAttribute('disabled', 'true');
        }

        // Enable Players Controls — 기본 재생 대상: 마스터 > 풀믹스 > 처리보컬
        const defaultMode = hasMastered ? 'mastered' : (hasFullMix ? 'fullmix' : 'processed');
        const defaultAudioMap = { mastered: audioMastered, fullmix: audioFullmix, processed: audioProcessed };
        currentAudio = defaultAudioMap[defaultMode];
        activeMode = defaultMode;

        modeBtnRaw.removeAttribute('disabled');
        modeBtnProcessed.removeAttribute('disabled');
        modeBtnRef.removeAttribute('disabled');
        masterPlayBtn.removeAttribute('disabled');
        globalSeek.removeAttribute('disabled');

        [modeBtnRaw, modeBtnProcessed, modeBtnRef, modeBtnFullmix, modeBtnFullmixRaw, modeBtnMastered]
            .forEach((b) => b.classList.remove('active'));
        ({ mastered: modeBtnMastered, fullmix: modeBtnFullmix, processed: modeBtnProcessed })[defaultMode].classList.add('active');

        // Update download button — 마스터 > 풀믹스 > 처리보컬 순으로 다운로드 우선
        const dlUrl = hasMastered ? data.audio_urls.full_mix_mastered : (hasFullMix ? data.audio_urls.full_mix : data.audio_urls.processed);
        const dlName = hasMastered ? 'mastered.wav' : (hasFullMix ? 'full_mix.wav' : 'processed_vocal.wav');
        downloadBtn.href = dlUrl;
        downloadBtn.setAttribute('download', dlName);
        downloadBtn.style.display = 'inline-flex';
        
        // Show EQ Chart & Tonal Error Card if EQ mode is enabled
        const chartCard = document.getElementById('chart-card');
        if (mixMode.includes('eq') && data.chart_data) {
            if (chartCard) chartCard.style.display = 'block';
            const tonalCardElem = document.getElementById('tonal-error-val-card');
            if (tonalCardElem && data.tonal_error !== undefined) {
                tonalCardElem.textContent = `${data.tonal_error.toFixed(2)} dB`;
            }
            renderChart(data.chart_data, data.compression_data);
        } else {
            if (chartCard) chartCard.style.display = 'none';
        }
        
        // Update 1176 Compressor Card
        const compData = data.compression_data;
        if (mixMode.includes('comp') && compData) {
            document.getElementById('comp-meter-card').style.display = 'block';
            
            const dynCardElem = document.getElementById('dynamics-error-val-card');
            if (dynCardElem && data.dynamics_error !== undefined) {
                dynCardElem.textContent = `${data.dynamics_error.toFixed(2)}`;
            }
            
            // e2e 엔진: PLR(= True Peak − Integrated LUFS) / legacy: 음절 피크 편차
            const isE2E = (data.engine === 'e2e');
            const srcVar = compData.src_dynamic_range || 0.0;
            const refVar = compData.ref_dynamic_range || 0.0;
            const shapedVar = compData.shaped_dynamic_range || 0.0;

            const setT = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };

            setT('comp-engine-label', isE2E ? '(Differentiable Peak Compressor · E2E 학습)'
                                            : '(1176-Style FET Compressor · 이진탐색)');
            // 지표명은 서버가 알려준 것을 따른다(크레스트 팩터 / PLR 전환 가능)
            const dynMetricName = compData.dyn_metric === 'plr'
                ? '다이내믹 레인지 (PLR = True Peak − Integrated LUFS)'
                : '다이내믹 레인지 (Crest Factor = True Peak − RMS)';
            setT('dyn-metric-title', isE2E ? dynMetricName : '음절 피크 균일도 (Syllable Peak Std)');
            // 다이내믹은 곡 전체가 아니라 하이라이트 구간 기준으로 맞춘다. 기준이 다르면
            // 곡 전체 PLR 과 값이 달라지므로, 무엇을 기준으로 잰 수치인지 명시한다.
            let dynNote = isE2E
                ? (compData.dyn_metric === 'plr' ? ' · ITU-R BS.1770-4' : ' · True Peak 는 BS.1770-4, RMS 는 단순 실효값')
                : ' · 음절 피크 dB의 표준편차';
            if (isE2E && compData.dyn_scope === 'highlight') {
                const w = compData.dyn_window_raw_s, wr = compData.dyn_window_ref_s;
                const secs = compData.dyn_window_sec || 15;
                dynNote += ` · <strong style="color:#00f2fe;">하이라이트 ${secs}초 구간 기준</strong>`
                    + (w && wr ? ` (내 보컬 ${w[0]}~${w[1]}s · 레퍼런스 ${wr[0]}~${wr[1]}s)` : '')
                    + (compData.shaped_dynamic_range_full !== undefined
                        ? ` · 참고: 곡 전체 PLR ${compData.shaped_dynamic_range_full.toFixed(2)} dB`
                        : '');
            }
            const dynNoteEl = document.getElementById('dyn-metric-note');
            if (dynNoteEl) dynNoteEl.innerHTML = dynNote;

            setT('syl-std-raw-val',  `${srcVar.toFixed(2)} dB`);
            setT('syl-std-ref-val',  `${refVar.toFixed(2)} dB`);
            setT('syl-std-proc-val', `${shapedVar.toFixed(2)} dB`);

            // LUFS 보조 표기 (e2e 만 제공)
            setT('lufs-raw-val',  compData.lufs_raw  !== undefined ? `${compData.lufs_raw} LUFS`  : '');
            setT('lufs-ref-val',  compData.lufs_ref  !== undefined ? `${compData.lufs_ref} LUFS`  : '');
            setT('lufs-proc-val', compData.lufs_proc !== undefined ? `${compData.lufs_proc} LUFS` : '');

            // 목표까지 남은 오차 (작을수록 좋음)
            const sylDiffElem = document.getElementById('syl-std-diff-val');
            if (sylDiffElem) {
                const gapBefore = Math.abs(srcVar - refVar);
                const gapAfter  = Math.abs(shapedVar - refVar);
                sylDiffElem.textContent = `${gapAfter.toFixed(2)} dB`;
                sylDiffElem.style.color = gapAfter < gapBefore ? '#4facfe' : '#ff6b6b';
                setT('comp-learn-note', `목표 대비 ${gapBefore.toFixed(2)} → ${gapAfter.toFixed(2)} dB`);
            }
            
            // 컴프레서 동작 정보
            const maxGRElem = document.getElementById('max-gr-val');
            if (maxGRElem) {
                const grArr = compData.gain_reduction_max;
                const grVal = Array.isArray(grArr) ? grArr[0] : (grArr || 0.0);
                maxGRElem.textContent = `${grVal.toFixed(1)} dB`;
            }
            
            setT('comp-thr-val', compData.threshold_db !== undefined
                 ? `${compData.threshold_db.toFixed(1)} dB` : '— dB');

            if (compData.ratio !== undefined) {
                const r = compData.ratio;
                const rLabel = r >= 999 ? '∞:1 (Limiter)' : `${r.toFixed(2)}:1`;
                setT('comp-method-val', `${isE2E ? 'Peak Detector' : '1176'} / ${rLabel}`);
            }

            // 어택/릴리즈: e2e 는 실제 사용값을 응답에서 받아 표시
            if (compData.attack_ms !== undefined) {
                setT('comp-ballistics-val', `${compData.attack_ms}ms / ${compData.release_ms}ms`);
            } else {
                setT('comp-ballistics-val', '3ms / Auto');
            }

            // Dynamic Range Footer (syllable std 값 그대로 표시)
            const srcDRElem    = document.getElementById('comp-src-dr');
            const refDRElem    = document.getElementById('comp-ref-dr');
            const shapedDRElem = document.getElementById('comp-shaped-dr');
            
            if (srcDRElem)    srcDRElem.textContent    = `${srcVar.toFixed(1)} dB`;
            if (refDRElem)    refDRElem.textContent    = `${refVar.toFixed(1)} dB`;
            if (shapedDRElem) shapedDRElem.textContent = `${shapedVar.toFixed(1)} dB`;
        } else {
            document.getElementById('comp-meter-card').style.display = 'none';
        }

        
        // Update Noise Gate Card (raw 전처리 결과 — mode 무관, 항상 표시)
        const gateData = data.gate_data;
        const gateCard = document.getElementById('gate-card');
        if (gateData) {
            gateCard.style.display = 'block';
            const badge = document.getElementById('gate-status-badge');
            const setG = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
            const fmt = (v) => (typeof v === 'number') ? v.toFixed(1) : '—';

            if (gateData.active) {
                badge.textContent = '● 적용됨 (ACTIVE)';
                badge.style.background = 'rgba(57, 255, 20, 0.12)';
                badge.style.border = '1px solid rgba(57, 255, 20, 0.4)';
                badge.style.color = '#39ff14';
                setG('gate-floor-val',     `${fmt(gateData.noise_floor_db)} dB`);
                setG('gate-thresh-val',    `${fmt(gateData.threshold_db)} dB`);
                setG('gate-snr-val',       `${fmt(gateData.snr_db)} dB`);
                setG('gate-reduction-val', `${fmt(gateData.reduction_db)} dB`);
                document.getElementById('gate-note').textContent =
                    `노이즈 플로어(${fmt(gateData.noise_floor_db)}dB) 위 ${fmt(gateData.threshold_db - gateData.noise_floor_db)}dB 지점에 스레숄드 자동 설정 → 구절 사이 배경 소음 제거.`;
            } else {
                badge.textContent = '○ 바이패스 (BYPASS)';
                badge.style.background = 'rgba(160, 168, 192, 0.12)';
                badge.style.border = '1px solid rgba(160, 168, 192, 0.4)';
                badge.style.color = '#a0a8c0';
                setG('gate-floor-val',     gateData.noise_floor_db !== undefined ? `${fmt(gateData.noise_floor_db)} dB` : '— dB');
                setG('gate-thresh-val',    '미적용');
                setG('gate-snr-val',       gateData.snr_db !== undefined ? `${fmt(gateData.snr_db)} dB` : '— dB');
                setG('gate-reduction-val', '0 dB');
                const reasonMap = { 'already clean': '배경이 이미 충분히 조용함 (원음 보존)', 'low SNR': '노이즈/보컬 분리 불가 (SNR 부족)', 'insufficient signal': '유효 신호 부족' };
                document.getElementById('gate-note').textContent =
                    `게이트 미적용 — ${reasonMap[gateData.reason] || '원음 보존'}.`;
            }
        } else if (gateCard) {
            gateCard.style.display = 'none';
        }

        // Update Reverb Meters & Status Card
        const reverbData = data.reverb_data;
        if (mixMode.includes('reverb') && reverbData && reverbData.active) {
            document.getElementById('reverb-card').style.display = 'block';
            
            const revCardElem = document.getElementById('reverb-error-val-card');
            if (revCardElem && data.reverb_error !== undefined) {
                revCardElem.textContent = `${data.reverb_error.toFixed(2)}`;
            }
            
            const isE2Erev = (data.engine === 'e2e');
            const isManual = (reverbMethod === 'manual') || !!reverbData.manual;
            const isMeasured = !isManual && (reverbMethod === 'measured');
            const srcLabel = (irSource === 'reference') ? '레퍼런스 보컬' : '반주';
            const setR = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };

            setR('reverb-rt60-val', `${reverbData.rt60.toFixed(2)}s`);
            const wetNow = (reverbData.wet || 0) * 100;
            setR('reverb-wet-val', `${wetNow.toFixed(1)}%`);

            if (isManual) {
                // 수동 모드: 사용자가 RT60 과 wet 을 직접 지정했다. 추정도 학습도 없다.
                setR('reverb-engine-label', '(수동 지정 · 추정/학습 없음)');
                setR('reverb-error-label', '잔향 감쇠 오차 (EDC, 참고용)');

                setR('rt60-badge', '사용자 지정');
                setR('reverb-edt-val', '');
                setR('reverb-rt60-src', '지수감쇠 IR 합성 · α = ln(1000)/(RT60·fs)');

                setR('reverb-wet-label', '잔향 혼합 비율 (Wet)');
                setR('wet-badge', '사용자 지정');
                setR('reverb-c80-val', '');
                setR('reverb-wet-note', '출력 = (1−wet)·원음 + wet·잔향');

                setR('band-badge', '');
                setR('reverb-band-val', '—');
                setR('reverb-band-note', '합성 리버브는 전대역 RT60 하나만 쓴다');

                setR('reverb-basis-label', '설정 방식');
                setR('reverb-sim-val', '수동');
                setR('reverb-ir-len', '');
                setR('reverb-method-note', '레퍼런스를 참조하지 않음 — IR 추정과 학습을 건너뜀');
            } else if (isMeasured) {
                // 측정 IR 모드: Rec-RIR 이 추정한 IR 을 그대로 컨볼브한다.
                // 리버브에는 학습 파라미터가 없다 — RT60 도 Wet 도 학습 대상이 아니다.
                const n1 = (v, unit, digits = 2) =>
                    (v === undefined || v === null) ? '—'
                        : `${v >= 0 && unit === ' dB' ? '+' : ''}${v.toFixed(digits)}${unit}`;

                setR('reverb-engine-label', '(Rec-RIR 블라인드 IR 추정 · 리버브 학습 없음)');
                // EDC 오차는 최적화 대상이 아니라 참고 지표다 (리버브 학습 파라미터 없음)
                setR('reverb-error-label', '잔향 감쇠 오차 (EDC, 참고용)');

                // ① 잔향 시간 — RT60(테일 전체) + EDT(체감 잔향감)
                setR('rt60-badge', 'IR에서 측정');
                setR('reverb-edt-val', `EDT ${n1(reverbData.edt, 's')} (체감 잔향)`);
                setR('reverb-rt60-src', `${srcLabel}의 IR · Schroeder EDC 기울기`);

                // ② 잔향 센드량 + IR 자체의 DRR / 명료도
                const drr = reverbData.drr_db;
                setR('reverb-wet-label', '잔향 센드량');
                setR('wet-badge', '매칭 강도');
                setR('reverb-wet-val', `${wetNow.toFixed(1)}%`);
                setR('reverb-c80-val',
                    `IR의 DRR ${n1(drr, ' dB', 1)} · C80 ${n1(reverbData.c80_db, ' dB', 1)}`);
                setR('reverb-wet-note', '드라이는 IR을 통과하지 않음 (고역 보존) · 잔향만 더함');

                // ③ 대역별 감쇠 — 공간의 음색. 고역/저역 비가 핵심이다.
                const ratio = reverbData.hf_lf_ratio;
                setR('reverb-band-val',
                    `${n1(reverbData.rt60_low, '')} / ${n1(reverbData.rt60_mid, '')} / ${n1(reverbData.rt60_high, '')}s`);
                setR('band-badge', '저 / 중 / 고');
                setR('reverb-band-note', !ratio ? '—'
                    : `고역/저역 ${ratio.toFixed(2)} · ${
                        ratio <= 0.7 ? '실제 방에 가까움' :
                        ratio <= 1.0 ? '고역이 늦게 죽음' : '고역이 저역보다 오래 남음 (비정상)'}`);

                // ④ 추정 방식
                setR('reverb-basis-label', '추정 방식');
                setR('reverb-sim-val', srcLabel);
                setR('reverb-ir-len',
                    `IR ${(reverbData.ir_ms || 0).toFixed(0)}ms · 분석 ${(reverbData.analysis_start_s || 0).toFixed(1)}s~`);
                setR('reverb-method-note', '잔향 스펙트럼 재구성으로 CTF 추정 → RIR 변환 (16kHz)');
            } else {
                setR('reverb-edt-val', '');
                setR('reverb-c80-val', '');
                setR('reverb-ir-len', '');
                setR('band-badge', '');
                setR('reverb-band-val', '—');
                setR('reverb-band-note', '합성 리버브는 전대역 RT60 하나만 쓴다');
                // 합성 리버브 폴백: 지수감쇠 IR 을 만들고 wet 만 학습한다.
                setR('reverb-engine-label', isE2Erev
                    ? '(RT60 블라인드 추출 + Wet E2E 학습)'
                    : '(Wasserstein 분포 매칭)');
                setR('reverb-error-label', '잔향 감쇠 오차 (EDC)');

                setR('rt60-badge', isE2Erev ? '추출·고정' : '학습');
                setR('reverb-rt60-src', isE2Erev
                    ? `레퍼런스 음절 감쇠 ${reverbData.n_segments || 0}구간에서 추출`
                    : '경사하강 최적화');

                setR('reverb-wet-label', '잔향 혼합 비율 (Wet)');
                setR('wet-badge', isE2Erev ? 'E2E 학습' : '학습');
                const wetInit = (reverbData.wet_init || 0) * 100;
                setR('reverb-wet-note', `초기값 ${wetInit.toFixed(1)}% → 학습 ${wetNow.toFixed(1)}%`);

                setR('reverb-basis-label', '추출 근거');
                if (isE2Erev) {
                    setR('reverb-sim-val', `${reverbData.n_segments || 0}개 구간`);
                    setR('reverb-method-note', '음절 사이 묵음의 에너지 포락선 회귀 → RT60');
                } else {
                    setR('reverb-sim-val', reverbData.similarity !== undefined
                         ? `${reverbData.similarity.toFixed(1)}%` : '—');
                    setR('reverb-method-note', '1D EMD Wasserstein Fit');
                }
            }
        } else {
            document.getElementById('reverb-card').style.display = 'none';
        }

        // 리버브 4조합 비교 플레이어
        renderReverbCompare(data.reverb_variants);

        // Update Mastering Card
        const masterData = data.mastering_data;
        const masterCard = document.getElementById('master-card');
        if (masterData) {
            masterCard.style.display = 'block';
            const setM = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
            setM('master-final-lufs',  `${masterData.final_lufs} LUFS`);
            setM('master-final-lufs2', `${masterData.final_lufs} LUFS`);
            setM('master-ref-lufs',    `${masterData.ref_lufs} LUFS`);
            setM('master-makeup',      `${masterData.makeup_db >= 0 ? '+' : ''}${masterData.makeup_db} dB`);
            setM('master-peak',        `${masterData.final_peak_db} / ${masterData.ceiling_db} dBFS`);
            if (masterData.eq_curve_x && masterData.eq_curve_y) {
                renderMasterEqChart(masterData.eq_curve_x, masterData.eq_curve_y);
            }
        } else if (masterCard) {
            masterCard.style.display = 'none';
        }

        // Success feedback
        processBtn.querySelector('.btn-text').textContent = '믹싱 완료!';
        setTimeout(() => {
            processBtn.removeAttribute('disabled');
            processBtn.querySelector('.btn-text').textContent = '믹스 시작하기';
            btnLoader.setAttribute('hidden', 'true');
        }, 2000);
        
    } catch (error) {
        alert('에러: ' + error.message);
        processBtn.removeAttribute('disabled');
        processBtn.querySelector('.btn-text').textContent = '믹스 시작하기';
        btnLoader.setAttribute('hidden', 'true');
    }
});

// 5. Chart.js Visualization
function renderChart(chartData, compData) {
    const ctx = document.getElementById('eqChart').getContext('2d');
    
    // Destroy previous chart if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }
    
    const xLabels = chartData.frequencies;
    const rawData = chartData.raw_envelope;
    const refData = chartData.ref_envelope;
    const procData = chartData.proc_envelope;
    
    // Filter curve variables
    const eqX = chartData.eq_curve_x;
    const eqY = chartData.eq_curve_y;
    
    const rawScatter = rawData.map((val, idx) => ({ x: xLabels[idx], y: val }));
    const refScatter = refData.map((val, idx) => ({ x: xLabels[idx], y: val }));
    const procScatter = procData.map((val, idx) => ({ x: xLabels[idx], y: val }));
    const eqScatter = eqY.map((val, idx) => ({ x: eqX[idx], y: val }));
    
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Applied EQ (dB)',
                    data: eqScatter,
                    borderColor: '#39ff14',
                    backgroundColor: 'rgba(57, 255, 20, 0.08)',
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0,
                    yAxisID: 'yEQ',
                    tension: 0.3
                },
                {
                    label: '내 보컬 (Raw)',
                    data: rawScatter,
                    borderColor: '#ff3366',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'yEnvelope',
                    tension: 0.2
                },
                {
                    label: '대상 보컬 (Reference)',
                    data: refScatter,
                    borderColor: '#00d2ff',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'yEnvelope',
                    tension: 0.2
                },
                {
                    label: '예상 매칭 보컬 (Processed)',
                    data: procScatter,
                    borderColor: '#9b51e0',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    yAxisID: 'yEnvelope',
                    tension: 0.2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: 'Frequency (Hz)',
                        color: '#a0a8c0'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#a0a8c0',
                        callback: function(value) {
                            const ticks = [200, 500, 1000, 2000, 5000, 10000];
                            if (ticks.includes(value)) {
                                return value >= 1000 ? (value / 1000) + 'k' : value;
                            }
                            return null;
                        }
                    },
                    // 표시 범위 = EQ 밴드 범위 = 손실의 평가 대역(200Hz~10kHz).
                    // 그 밖은 EQ 가 건드리지도, 손실이 보지도 않으므로 그릴 이유가 없다.
                    // (200Hz 아래는 분리 과정에서 보컬 저역이 소실되고, 10kHz 위는 mp3
                    //  코덱 절벽이 그대로 보여 둘 다 오해를 부른다.)
                    min: 200,
                    max: 10000
                },
                yEnvelope: {
                    type: 'linear',
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Magnitude Envelope (dB)',
                        color: '#a0a8c0'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#a0a8c0'
                    },
                    // 고정 범위를 두지 않는다. 포락선은 '총에너지로 나눈 값(항상 음수)'이
                    // 아니라 **평가 대역 평균 dB 를 뺀 값**이라 0 을 넘나든다(실측 -22 ~ +17 dB).
                    // 예전의 min:-60/max:0 을 그대로 두면 곡선이 통째로 축 밖으로 잘린다.
                    grace: '5%'
                },
                yEQ: {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Equalizer Gain (dB)',
                        color: '#e0e0e0'
                    },
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        color: '#e0e0e0'
                    },
                    min: -15,
                    max: 15
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f5f6fa',
                        boxWidth: 12
                    }
                },
                tooltip: {
                    callbacks: {
                        title: function(context) {
                            const freq = parseFloat(context[0].raw.x).toFixed(1);
                            return `주파수: ${freq} Hz`;
                        },
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += parseFloat(context.parsed.y).toFixed(1) + ' dB';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// Mastering EQ curve chart
function renderMasterEqChart(eqX, eqY) {
    const canvas = document.getElementById('masterEqChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (masterChartInstance) masterChartInstance.destroy();

    const data = eqY.map((val, idx) => ({ x: eqX[idx], y: val }));
    masterChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Master EQ Gain (dB)',
                data: data,
                borderColor: '#39ff14',
                backgroundColor: 'rgba(57, 255, 20, 0.08)',
                borderWidth: 2,
                fill: true,
                pointRadius: 0,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f5f6fa', boxWidth: 12 } } },
            scales: {
                x: {
                    type: 'logarithmic',
                    min: 20, max: 20000,
                    title: { display: true, text: 'Frequency (Hz)', color: '#a0a8c0' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: {
                        color: '#a0a8c0',
                        callback: (v) => {
                            const t = [50, 100, 500, 1000, 5000, 10000, 20000];
                            return t.includes(v) ? (v >= 1000 ? (v / 1000) + 'k' : v) : null;
                        }
                    }
                },
                y: {
                    title: { display: true, text: 'Gain (dB)', color: '#a0a8c0' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#a0a8c0' },
                    min: -10, max: 10
                }
            }
        }
    });
}


// ---- 리버브 방식 / IR 소스: 그룹 내 배타 선택 ----
function bindExclusiveToggle(ids) {
    const btns = ids.map(id => document.getElementById(id)).filter(Boolean);
    btns.forEach(btn => btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }));
}
bindExclusiveToggle(['rmode-measured', 'rmode-synth', 'rmode-manual']);
bindExclusiveToggle(['rsrc-instrumental', 'rsrc-reference']);

// ---- 리버브 방식에 따라 관련 컨트롤만 보여준다 ----
// 수동 모드에서는 (a) 측정 대상·4조합 비교가 의미 없고, (b) 잔향량 슬라이더는 wet 과
// 역할이 겹쳐 노브가 두 개가 되므로 숨긴다. 서버도 수동일 때 reverb_amount 를 1.0 으로
// 고정하므로, 숨기지 않으면 화면 값과 실제 동작이 어긋난다.
function syncReverbModeUI() {
    const manual = selectedValue(['rmode-measured', 'rmode-synth', 'rmode-manual'], 'manual') === 'manual';
    const show = (id, on) => {
        const el = document.getElementById(id);
        if (el) el.style.display = on ? '' : 'none';
    };
    show('auto-reverb-opts', !manual);
    show('manual-reverb-opts', manual);
    show('reverb-amount-group', !manual);
}
['rmode-measured', 'rmode-synth', 'rmode-manual'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', syncReverbModeUI);
});

const manualRt60Slider = document.getElementById('manual-rt60');
const manualRt60Val = document.getElementById('manual-rt60-val');
const manualWetSlider = document.getElementById('manual-wet');
const manualWetVal = document.getElementById('manual-wet-val');
if (manualRt60Slider) manualRt60Slider.addEventListener('input', (e) => {
    manualRt60Val.textContent = `${parseFloat(e.target.value).toFixed(2)} s`;
});
if (manualWetSlider) manualWetSlider.addEventListener('input', (e) => {
    manualWetVal.textContent = `${Math.round(e.target.value * 100)}%`;
});
syncReverbModeUI();

function selectedValue(ids, fallback) {
    for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.classList.contains('active')) return el.dataset.value;
    }
    return fallback;
}


// ---- 리버브 4조합 비교 플레이어 렌더 ----
function renderReverbCompare(variants) {
    const card = document.getElementById('reverb-compare-card');
    const list = document.getElementById('reverb-compare-list');
    if (!card || !list) return;
    const keys = variants ? Object.keys(variants) : [];
    if (keys.length === 0) { card.style.display = 'none'; return; }

    const LABEL = {
        measured_instrumental: '측정 IR · 반주',
        measured_reference:    '측정 IR · 레퍼런스',
        synth_instrumental:    '합성 RT60 · 반주',
        synth_reference:       '합성 RT60 · 레퍼런스',
    };
    const order = ['measured_instrumental', 'measured_reference', 'synth_instrumental', 'synth_reference'];

    list.innerHTML = '';
    order.filter(k => variants[k]).forEach(k => {
        const v = variants[k];
        const row = document.createElement('div');
        row.style.cssText = 'margin-bottom:14px; padding:12px; background:rgba(255,255,255,0.03);'
                          + 'border:1px solid rgba(255,255,255,0.08); border-radius:8px;';
        const rt = v.rt60 ? `RT60 ${v.rt60.toFixed(2)}s · ` : '';
        row.innerHTML =
            `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
               <strong style="color:#4facfe;">${LABEL[k] || k}</strong>
               <span style="font-size:0.78rem; color:#a0a8c0;">${rt}wet ${(v.wet * 100).toFixed(0)}% · 감쇠구간 ${v.n_segments}</span>
             </div>
             <audio controls preload="none" style="width:100%;" src="${v.url}"></audio>`;
        list.appendChild(row);
    });
    card.style.display = 'block';
}
