# boson.py
import shutil
import sys
from pathlib import Path
from loguru import logger

ROOT = Path(__file__).resolve().parent
HIGGS_ROOT = ROOT / "faster-higgs-audio"

sys.path.insert(0, str(HIGGS_ROOT))

import warnings
import torch

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from patches.generation import HiggsAudioModelClient

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model and tokenizer names
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

def sync_voice_prompts():
    root = Path(__file__).resolve().parent
    src = root / "voices"
    dst = root / "faster-higgs-audio" / "examples" / "voice_prompts"

    dst.mkdir(parents=True, exist_ok=True)

    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)

def get_device(device_arg=None):
    """Determine the best device to use."""
    if device_arg == "cpu":
        return "cpu"
    elif device_arg == "cuda":
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
    elif device_arg == "auto" or device_arg is None:
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    else:
        return device_arg


def initialize_synthesization():

    device = get_device("auto")
    device_id = None if device == "cpu" else int(device.split(":")[-1])

    # audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=device)
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=get_device("cpu"))

    # To prevent multiple loadings, run this once
    model_client = HiggsAudioModelClient(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer=audio_tokenizer,
        device_id=device_id,
        max_new_tokens=2048, # 378, # $4096 / 8,
        use_static_kv_cache=False,
        use_quantization=True,
        quantization_bits=8,
    )

    return model_client, audio_tokenizer

