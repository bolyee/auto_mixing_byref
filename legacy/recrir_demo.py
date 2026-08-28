# Companion to inference.py: saves every signal the model produces, not just
# the RIR, so the result can be judged by ear.
#
#   output/rir/x.wav        estimated room impulse response  (same as inference.py)
#   output/dereverb/x.wav   dereverberated speech            (model's y_spch)
#   output/denoised/x.wav   denoised but still reverberant   (model's y_rev)
#   output/input/x.wav      the amplitude-normalised input, for A/B comparison
#
# With --dry the estimated RIR is convolved onto an anechoic file, which
# re-creates the room on new audio:
#
#   output/reapplied/x.wav

import os
from glob import glob
from pathlib import Path

import toml
import torch
import torchaudio
from jsonargparse import ArgumentParser
from tqdm import tqdm

from trainer_inferencer.utils import initialize_module


def load_model(model_cfg, ckpt_path, device):
    model = initialize_module(model_cfg["path"], model_cfg["args"])
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = {}
    for k, v in ckpt["model"].items():
        if any(x in k for x in ["ops", "params"]):
            continue
        state_dict[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def norm_save(TF, wav, path, sr):
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.95
    TF.save_wav(wav.cpu(), path, sr)


@torch.no_grad()
def main(
    input_path: str,
    output_path: str,
    model: dict,
    EM_algo: dict,
    acoustic: dict,
    ckpt: str,
    device: str,
    dry: str = None,
    **kwargs,
):
    fpath_input = sorted(glob(f"{input_path}/**/*.flac", recursive=True)) + sorted(
        glob(f"{input_path}/**/*.wav", recursive=True)
    )
    if not fpath_input:
        raise SystemExit(f"no .wav/.flac found under {input_path}")

    TF = initialize_module(acoustic["path"], acoustic["args"])
    sr = TF.sr
    mymodel = load_model(model, ckpt, device)
    pim = initialize_module(EM_algo["path"], EM_algo["args"])

    dry_wav = None
    if dry is not None:
        dry_wav = TF.load_wav(dry, sr).to(device)

    for fpath in tqdm(fpath_input):
        name = os.path.basename(fpath)

        obs_wav = TF.load_wav(fpath, sr).to(device)
        obs_normed = TF.norm_amplitude(obs_wav.squeeze())
        obs = TF.stft(obs_normed, "complex")

        outputs = mymodel(TF.preprocess(obs.unsqueeze(0).unsqueeze(1)))
        y_spch, _, y_rev = outputs

        n_sample = obs_wav.squeeze().shape[-1]
        dereverb = TF.istft(TF.postprocess(y_spch).squeeze(), "complex", n_sample)
        denoised = TF.istft(TF.postprocess(y_rev).squeeze(), "complex", n_sample)

        # pim owns the CTF -> RIR maths, but init_seg would re-run the network on
        # the exact same input. Hand it the outputs we already have instead.
        rir = pim.process(obs_wav, lambda *a, **kw: outputs, TF, device)

        norm_save(TF, obs_normed, os.path.join(output_path, "input", name), sr)
        norm_save(TF, dereverb, os.path.join(output_path, "dereverb", name), sr)
        norm_save(TF, denoised, os.path.join(output_path, "denoised", name), sr)
        norm_save(TF, rir, os.path.join(output_path, "rir", name), sr)

        if dry_wav is not None:
            wet = torchaudio.functional.fftconvolve(
                dry_wav.squeeze().cpu(), rir.squeeze().cpu(), mode="full"
            )
            norm_save(TF, wet, os.path.join(output_path, "reapplied", name), sr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("-i", "--input_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-d", "--device", required=False, type=str, default="cpu")
    parser.add_argument(
        "--dry",
        required=False,
        type=str,
        default=None,
        help="anechoic wav to convolve with each estimated RIR",
    )

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    config = toml.load(Path(args.config).expanduser().absolute().as_posix())
    main(**args, **config)

    """
    usage:
    python demo.py -c config/Rec-RIR.toml --ckpt ckpt/epoch35.tar -i input -o output -d cpu [--dry anechoic.wav]
    """
