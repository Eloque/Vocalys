# Create the folders and synthesize the audio.
import json
import os

import torch
from loguru import logger

from boson import initialize_synthesization, sync_voice_prompts, get_device
from worker import generate_context_per_voice, synthesization_loop

# Prevent fragmentation, all little bits help
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

def main():

    verify()

def verify():

    logger.info(f"Starting Vocalis Verification - Synthesizing Audio")
    client_model, tokenizer = initialize_synthesization()

    print(torch.cuda.is_available())
    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)

    # Try to load voices
    try:
        # raise Exception("test")
        voices = json.load(open("./voices/voices.json"))
        voices = voices["voices"]
        sync_voice_prompts()
    except:
        logger.info("No voices found, using default voice")
        voices = list()
        voices.append({"name": "Default", "style": None})

    logger.info(f"Creating context for voices")

    # generate the contexts once, to save time later
    generate_context_per_voice(voices, tokenizer)

    # check if the output folder exists, if not create it
    filename = "Verification.wav"

    # The main text
    result, chunked_size_success = synthesization_loop(client_model,
                                                       "The Frost settles on the town of Frosthaven",
                                                       voices[0],
                                                       filename)

    if result:
        logger.info(f"Audio saved to {filename}")


if __name__ == "__main__":
    main()