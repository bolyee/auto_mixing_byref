# Convolve a dry signal with an impulse response. Standalone: no model, no
# checkpoint, no 16 kHz restriction.
#
#   python apply_ir.py --dry vocal.wav --ir output/rir/x.wav -o wet.wav
#
# --ir may also be a directory, in which case every IR inside it is applied and
# the results are written next to each other under -o.
#
# The dry file's sample rate and channel count are preserved; the IR is
# resampled to match, so a 16 kHz estimated RIR can be used on 44.1 kHz audio.

import os
from glob import glob
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from jsonargparse import ArgumentParser


def load(path):
    wav, sr = torchaudio.load(path, channels_first=True)
    return wav, sr


def apply_ir(dry, ir, mix, normalize):
    """dry: [C, N], ir: [1, M] -> [C, N + M - 1]"""
    wet = torch.stack(
        [torchaudio.functional.fftconvolve(ch, ir[0], mode="full") for ch in dry]
    )
    if mix < 1.0:
        padded_dry = torch.nn.functional.pad(dry, (0, wet.shape[-1] - dry.shape[-1]))
        wet = mix * wet + (1.0 - mix) * padded_dry
    if normalize:
        peak = wet.abs().max()
        if peak > 0:
            wet = wet / peak * 0.95
    return wet


def main():
    parser = ArgumentParser()
    parser.add_argument("--dry", required=True, type=str, help="dry input wav")
    parser.add_argument("--ir", required=True, type=str, help="IR wav file or directory")
    parser.add_argument("-o", "--output", required=True, type=str, help="output wav or directory")
    parser.add_argument("--mix", type=float, default=1.0, help="wet ratio, 0..1 (default 1.0 = fully wet)")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="skip peak normalisation")
    args = parser.parse_args()

    dry, sr_dry = load(args.dry)

    ir_paths = (
        sorted(glob(os.path.join(args.ir, "*.wav")) + glob(os.path.join(args.ir, "*.flac")))
        if os.path.isdir(args.ir)
        else [args.ir]
    )
    if not ir_paths:
        raise SystemExit(f"no IR files found in {args.ir}")

    multi = len(ir_paths) > 1 or os.path.isdir(args.output)
    if multi:
        os.makedirs(args.output, exist_ok=True)

    for ir_path in ir_paths:
        ir, sr_ir = load(ir_path)
        if ir.shape[0] > 1:
            ir = ir.mean(0, keepdim=True)
        if sr_ir != sr_dry:
            ir = torchaudio.functional.resample(ir, sr_ir, sr_dry)

        wet = apply_ir(dry, ir, args.mix, args.normalize)

        if multi:
            out = os.path.join(args.output, f"{Path(args.dry).stem}__{Path(ir_path).stem}.wav")
        else:
            out = args.output
            os.makedirs(Path(out).parent, exist_ok=True) if Path(out).parent != Path("") else None

        sf.write(out, wet.T.numpy(), sr_dry)
        print(f"{Path(ir_path).name}  ->  {out}   ({wet.shape[-1] / sr_dry:.2f}s @ {sr_dry} Hz)")


if __name__ == "__main__":
    main()
