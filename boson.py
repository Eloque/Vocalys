# boson.py
import time

import warnings
import chunker
import torch

import soundfile as sf

from boson_multimodal.data_types import ChatMLSample, Message
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from examples.generation import HiggsAudioModelClient, prepare_generation_context, prepare_chunk_text, normalize_chinese_punctuation

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Model and tokenizer names
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

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


def main():
    parser = argparse.ArgumentParser(
        description="Boson – Vocalis audio generation using HiggsAudio model"
    )

    parser.add_argument(
        "voice",
        help="Voice sample name to use (e.g. 'voice', 'chris', 'narrator_dark')"
    )

    args = parser.parse_args()

    voice = args.voice
    samples = [
        # Anchor / tonal conditioning sample (generate first, do NOT trim)
        "Take a steady breath. The cold waits for no one.",

        # Follow-ups (generate separately, appended after anchor in playback)
        "The air is thin, and it settles softly in your lungs.",
        "Each step carries weight, and the mountain remembers them all.",
        "Frost gathers along the edges of silence.",
        "Somewhere in the distance, something is watching.",
        "The wind moves through stone and bone alike.",
        "The world is vast, indifferent, and older than your fear."
    ]

    print(f"Using voice sample: {voice}")

    text = ""
    for sample in samples:
        text += sample + "\n"
        print(f"Sample ({len(sample)} chars): {sample}...")

    print()

    chunked_text = chunker.chunk_text(text)

    device = "cuda"
    audio_tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=device)

    device = get_device("auto")
    device_id = None if device == "cpu" else int(device.split(":")[-1])

    model_client = HiggsAudioModelClient(
        model_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer="bosonai/higgs-audio-v2-tokenizer",
        device_id=device_id,
        max_new_tokens=4096,
        use_static_kv_cache=False,
        use_quantization=True,
        quantization_bits=4,
    )

    speaker_tags = []

    messages, audio_ids = prepare_generation_context(
        scene_prompt="",
        ref_audio=voice,
        ref_audio_in_system_message=True,
        audio_tokenizer=audio_tokenizer,
        speaker_tags=speaker_tags,
    )

    # messages[1] = Message(
    #     role="system",
    #     content=(
    #         # "You are a deep, sonorous narrator. "
    #         # "Your voice is cold, restrained, and heavy with foreboding. "
    #         # "Each sentence carries weight. "
    #         # "Pause slightly between phrases. "
    #         # "Emotion is subtle but unmistakable."
    #         "You are a low, warm narrator."
    #         "Speech is steady, controlled, and deliberate."
    #         "Emotion is restrained."
    #         "Consonants are soft."
    #         "Pacing is moderate."
    #     ),
    # )

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=2,
        temperature=0.65,
        top_k=80,
        top_p=0.97,
        ras_win_len=4 + 20,
        ras_win_max_num_repeat=3 + 1,
        seed=1001,
    )

    sf.write("scen0.wav", concat_wv, sr)
    end_time = time.time()
    print(f"[Vocalis] Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
