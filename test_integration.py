import os
import numpy as np
import soundfile as sf
from pipeline import match_e2e

def run_test():
    print("Running integration test for DDSP Auto-EQ...")
    
    # 1. Create dummy audio signals
    sr = 44100
    duration = 3.0 # seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Raw signal: White noise + fundamental frequency
    raw_signal = 0.2 * np.random.randn(len(t)) + 0.3 * np.sin(2 * np.pi * 440.0 * t)
    # Reference signal: White noise + fundamental frequency with high-frequency boost (highpass filtered-like)
    ref_signal = 0.05 * np.random.randn(len(t)) + 0.3 * np.sin(2 * np.pi * 440.0 * t)
    # Boost high frequencies in ref_signal manually using a simple filter
    ref_signal = np.convolve(ref_signal, [0.5, -0.5], mode='same') # high pass effect
    
    # Normalize peaks to prevent clipping
    raw_signal /= np.max(np.abs(raw_signal)) + 1e-8
    ref_signal /= np.max(np.abs(ref_signal)) + 1e-8
    
    raw_path = "raw_test.wav"
    ref_path = "ref_test.wav"
    out_path = "out_test.wav"
    
    # Save dummy signals
    sf.write(raw_path, raw_signal, sr)
    sf.write(ref_path, ref_signal, sr)
    
    try:
        # 2. Run match_e2e
        print("Running match_e2e algorithm...")
        result = match_e2e(
            raw_path=raw_path,
            ref_path=ref_path,
            out_path=out_path,
            num_bands=5,
            match_amount=1.0,
            match_volume=True,
            comp_amount=0.8,
            n_steps=5,          # 스모크 테스트라 수렴이 아니라 동작만 확인
            verbose=False,
        )
        
        # 3. Assertions
        assert result["status"] == "success", "Status should be success"
        assert os.path.exists(out_path), "Output file should be created"
        
        # E2E 체인은 리버브가 스테레오 RIR 이라 항상 [T, 2] 로 저장된다
        y_out, sr_out = sf.read(out_path)
        assert y_out.ndim == 2 and y_out.shape[1] == 2, "E2E output should be stereo"
        assert np.isfinite(y_out).all(), "Output should not contain NaN/Inf"
        
        chart_data = result["chart_data"]
        expected_keys = ["frequencies", "raw_envelope", "ref_envelope", "proc_envelope", "eq_curve_x", "eq_curve_y", "bands_x", "bands_y"]
        for key in expected_keys:
            assert key in chart_data, f"Chart data should contain '{key}'"
            assert len(chart_data[key]) > 0, f"Chart data key '{key}' should not be empty"
            
        # Assert compression data
        assert "compression_data" in result, "Result should contain 'compression_data'"
        comp_data = result["compression_data"]
        assert len(comp_data["gain_reduction_max"]) == 1, "Should have 1 gain reduction max value"
        assert "src_dynamic_range" in comp_data, "Should have src_dynamic_range"
        assert "ref_dynamic_range" in comp_data, "Should have ref_dynamic_range"
        assert "shaped_dynamic_range" in comp_data, "Should have shaped_dynamic_range"
        
        # Assert average errors
        assert "match_error" in result, "Result should contain 'match_error'"
        assert "tonal_error" in result, "Result should contain 'tonal_error'"
        assert "dynamics_error" in result, "Result should contain 'dynamics_error'"
        match_err = result["match_error"]
        tonal_err = result["tonal_error"]
        dyn_err = result["dynamics_error"]
        assert match_err >= 0.0, "Match error should be positive"
        assert tonal_err >= 0.0, "Tonal error should be positive"
        assert dyn_err >= 0.0, "Dynamics error should be positive"
            
        print("✓ EQ & Crest Factor Shaping completed successfully!")
        print(f"✓ Dynamic Range: src={comp_data['src_dynamic_range']:.1f}dB → ref={comp_data['ref_dynamic_range']:.1f}dB → shaped={comp_data['shaped_dynamic_range']:.1f}dB")
        print(f"✓ Max Gain Reduction: {comp_data['gain_reduction_max'][0]:.2f} dB")
        print(f"✓ Combined Error: {match_err:.2f} dB")
        
        # Clean up output file
        if os.path.exists(out_path):
            os.remove(out_path)
            
        # 4. Run match_e2e in reverb mode
        print("\nRunning match_e2e algorithm in 'reverb' mode...")
        result_rev = match_e2e(
            raw_path=raw_path,
            ref_path=ref_path,
            out_path=out_path,
            num_bands=5,
            match_amount=1.0,
            match_volume=True,
            comp_amount=0.8,
            reverb_amount=0.9,
            mode="reverb",
            n_steps=5,
            verbose=False,
        )
        
        assert result_rev["status"] == "success", "Reverb mode status should be success"
        assert os.path.exists(out_path), "Reverb mode output file should be created"
        
        # Verify output is stereo for 'reverb' mode
        y_rev, sr_rev = sf.read(out_path)
        assert y_rev.ndim == 2 and y_rev.shape[1] == 2, "Reverb output should be stereo (T x 2 array)"
        
        assert "reverb_data" in result_rev, "Result should contain 'reverb_data'"
        reverb_data = result_rev["reverb_data"]
        for key in ("rt60", "wet", "active", "n_segments"):
            assert key in reverb_data, f"reverb_data should contain '{key}'"

        # 이 테스트 신호는 3초짜리 연속 노이즈+사인이라 **잔향이 전혀 없다**.
        # 따라서 올바른 동작은 둘 중 하나다:
        #   (a) 리버브가 자동 비활성된다, 또는
        #   (b) 추정 RT60 이 0 에 가깝다 (= 없는 잔향을 만들어내지 않는다)
        # 이 단언이 실패하면 IR 추정기가 드라이 신호에 잔향을 환각한다는 뜻이다.
        DRY_RT60_MAX = 0.3
        if reverb_data["active"]:
            assert reverb_data["rt60"] >= 0.0, "RT60 should be non-negative when active"
            assert reverb_data["rt60"] < DRY_RT60_MAX, (
                f"드라이 합성 신호인데 RT60 {reverb_data['rt60']:.2f}s 가 추정됐다 "
                f"(허용 {DRY_RT60_MAX}s). IR 추정기가 없는 잔향을 만들어내고 있다."
            )
            print(f"✓ Estimated RT60: {reverb_data['rt60']:.2f} s (dry signal, under {DRY_RT60_MAX} s)")
        else:
            print("✓ Reverb gracefully disabled (no reverb in synthetic signal)")

        print("✓ Reverb stage completed successfully!")
        print(f"✓ Output file is stereo: {y_rev.shape}")
        
        print("\n🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULY!")
        
    finally:
        # Clean up files
        for p in [raw_path, ref_path, out_path]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Cleaned up {p}")

if __name__ == "__main__":
    run_test()
