# Create the folders and synthesize the audio.
import datetime
import json
import os
import time

import torch

from loguru import logger
from enum import Enum, auto

import chunker
from boson import initialize_synthesization, sync_voice_prompts, get_device
from examples.generation import prepare_generation_context
import soundfile as sf

# Prevent fragmentation, all little bits help
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

class PageType(Enum):
    TITLE = auto()
    SCENARIO = auto()
    CONTINUED_SCENARIO = auto()
    UNKNOWN = auto()

def log_cuda_memory():
    """Log CUDA memory allocation and reservation in MB."""
    allocated_mb = torch.cuda.memory_allocated() / 1024 ** 2
    reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
    logger.info(f"CUDA Memory - Allocated: {allocated_mb:.2f} MB, Reserved: {reserved_mb:.2f} MB")

def generate_context_per_voice(voices, tokenizer):

    logger.info("Generating context for voices")

    for voice in voices:

        logger.info(f'Generating for voice: {voice["name"]}')

        # check if a voice has a chunk size, if it does,
        # use it as the initial chunk size for synthesis,
        # otherwise use the default chunk size
        max_chunk_size = 600

        if "chunk_size" in voice and voice["chunk_size"] is not None:
            pass
        else:
            voice["chunk_size"] = max_chunk_size

        if voice["name"] != "Default":
            messages, audio_ids = prepare_generation_context(
                scene_prompt=voice["style"],
                ref_audio=voice["name"],
                ref_audio_in_system_message=True,
                audio_tokenizer=tokenizer,
                speaker_tags=[]
            )
        else:
            messages, audio_ids = prepare_generation_context(
                scene_prompt="",
                ref_audio=None,
                ref_audio_in_system_message=False,
                audio_tokenizer=tokenizer,
                speaker_tags=[]
            )

            voice["temperature"] = 0.65
            voice["top_k"] = 80
            voice["top_p"] = 0.97

        voice["messages"] = messages
        voice["audio_ids"] = audio_ids

def synthesize_audio(model_client, text, voice, filename, max_chunk_size=120):
    """
    Generate speech audio for the given text using a prepared voice configuration.

    Args:
        model_client: Initialized audio generation client used to run inference.
        tokenizer: Audio/token context object required by the generation pipeline.
        text (str): Input text to synthesize.
        voice (dict): Voice configuration containing generation settings and
            prepared context such as prompt messages, audio IDs, and optional primer.
        filename (str): Output path for the generated audio file.
        max_chunk_size (int, optional): Maximum size used when splitting text into
            chunks for generation. Defaults to 400.

    Returns:
        bool: True if audio was synthesized and written, False if synthesis was
        skipped because the output file already existed.

    Notes:
        - Assumes the voice context has already been prepared before calling.
    """

    # check if the file already exists, if it does, skip the synthesis
    if os.path.exists(filename):
        logger.info(f"File {filename} already exists, skipping synthesis.")
        return False

    if voice == "None":
        voice = None

    logger.info(f"File {filename} does not exist, missing {voice['name']} voice sample, synthesizing...")

    # Chunk the text into smaller pieces based on the max_chunk_size
    chunked_text = chunker.chunk_text(text, max_chunk_size)
    primed  = False

    # check if voice has a primer, if it does, add it to the beginning of the chunked text
    if "primer" in voice and voice["primer"] is not None:
        chunked_text.insert(0, voice["primer"])
        primed = True

    messages = voice["messages"]
    audio_ids = voice["audio_ids"]

    ras_win_len = 32
    ras_win_max_num_repeat = 4
    generation_chunk_buffer_size = 3

    temperature = voice["temperature"]
    top_k = voice["top_k"]
    top_p = voice["top_p"]

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=generation_chunk_buffer_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=1001001,
        primed=primed
    )

    sf.write(filename, concat_wv, sr)
    torch.cuda.empty_cache()

    return True

def synthesization_loop(client_model,text, voice, filename):

    """
    Attempt audio synthesis, reducing chunk size after out-of-memory
    failures until synthesis succeeds or no smaller chunk size remains.

    Args:
        max_chunk_size (int): Initial chunk size to use for synthesis.
        client_model: Initialized audio generation client used to run inference.
        text (str): Input text to synthesize.
        voice (dict): Voice configuration containing generation settings and
            prepared context such as prompt messages, audio IDs, and optional primer.
        filename (str): Output path for the generated audio file.

    Returns:
        tuple[bool, int]: Whether synthesis succeeded and the final chunk size used.
    """

    synthesizing = True
    result = False

    max_chunk_size = voice["chunk_size"]

    while synthesizing:
        oom = False

        try:
            start = time.perf_counter()
            logger.info(f"Synthesizing with chunk size: {max_chunk_size}")
            log_cuda_memory()

            result = synthesize_audio(client_model, text, voice, filename, max_chunk_size=max_chunk_size)

            elapsed = time.perf_counter() - start
            logger.info(f"{elapsed:.2f}s to synthesize {filename}")

            synthesizing = False

        except torch.OutOfMemoryError:
            logger.info("OOM on clip:", filename)
            log_cuda_memory()
            oom = True

        except Exception as e:
            logger.error("Error on clip:", filename)
            logger.error(e)

        if oom:
            logger.info("Leaving exception handler")
            log_cuda_memory()

            if max_chunk_size > 100:
                max_chunk_size -= 20
                logger.info(f"Retrying with smaller chunk size: {max_chunk_size}")
            else:
                logger.info(f"Error synthesizing audio for {filename}: OOM, no chunk size left, skipping.")
                synthesizing = False

    return result, max_chunk_size

def main():

    # scenarios()
    # sections()
    # sampler()

    introductions()

def introductions():

    logger.info(f"Starting Vocalis - Synthesizing Audio")
    # Load models
    client_model, tokenizer = initialize_synthesization()

    print(torch.cuda.is_available())
    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)

    voices = json.load(open("./voices/voices.json"))
    voices = voices["voices"]
    sync_voice_prompts()

    generate_context_per_voice(voices, tokenizer)

    # only do the the first two items from voices
    voices = voices[:2]

    # for each voice, create a folder for it
    for voice in voices:
        # Creat the folder that is needed for the voice
        voice_folder = os.path.join("./output/introductions", voice["name"])
        if not os.path.exists(voice_folder):
            os.makedirs(voice_folder)

        filename = f"Introduction-{voice['name']}.wav"
        filename = os.path.join(voice_folder, filename)

        # The main text
        result, chunked_size_success = synthesization_loop(client_model,
                                                           voice["introduction"],
                                                           voice,
                                                           filename)

        if result:
            print(f"Audio file saved to {filename}")


def sections():

    logger.info(f"Starting Vocalis - Synthesizing Audio")
    # Load models
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

    input_file = "./input/scenarios/sections_book.json"
    book = json.load(open(input_file))

    # check if the output folder exists, if not create it
    output_folder = "./output/sections"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sections_to_run = ["106.4"]

    for entry in book[:]:

        try:

            if entry["number"] not in sections_to_run:
                continue

            section_folder = os.path.join(output_folder, f"{entry['number']}")

            # Write the manifest file
            manifest_file = os.path.join(section_folder, "manifest.json")

            # Check if the manifest file already exists, if it does, read it and update it with the new manifest, otherwise create a new manifest file
            if os.path.exists(manifest_file):
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            else:
                manifest = {
                    "section": {
                        "number": entry["title"],
                        "text" : ""
                    }
                }

            for section in entry["sections"]:
                if section["header"] == "Continuation" or section["header"] == "Conclusion":
                    manifest["section"]["text"] = section["text"]

            for voice in voices:
                # Creat the folder that is needed for the voice
                voice_folder = os.path.join(section_folder, voice["name"])
                if not os.path.exists(voice_folder):
                    os.makedirs(voice_folder)

                filename = "Section.wav"
                filename = os.path.join(voice_folder, filename)

                # The main text
                result, chunked_size_success = synthesization_loop(client_model,
                                                                   manifest["section"]["text"] ,
                                                                   voice,
                                                                   filename)

                if result:
                    audio = {
                        "voice": voice["name"],
                        "file": f"{voice['name']}/Section.wav",
                        "chunk_size": chunked_size_success,
                        "creation_time": datetime.datetime.now().isoformat()
                    }

                    # check if the manifest scenario has an audio list
                    if "audio" not in manifest["section"]:
                        manifest["section"]["audio"] = list()

                    manifest["section"]["audio"].append(audio)

                    with open(manifest_file, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(e)

#3
def sampler():

    logger.info(f"Starting Vocalis - Synthesizing Audio")
    # Load models
    client_model, tokenizer = initialize_synthesization()

    print(torch.cuda.is_available())
    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)

    logger.info(f"Creating context for voice")

    elarion_prompt = """The narrator is Elaria, a female voice.

Her voice is soft, warm, and gently resonant.
It is clearly feminine, with a smooth and slightly airy quality.

Her tone is calm and controlled, but never heavy or deep.
Avoid a masculine or overly low-pitched voice.

Pacing is slow and natural.
Sentences flow smoothly with gentle pauses.

Consonants are soft and rounded.
Sibilance is light and controlled.

Emotion is subtle and restrained.
No theatrical delivery, no sharpness.

The voice should feel warm, calm, and slightly ethereal, like a quiet and confident female storyteller."""

    voice = dict()
    voice["name"] = "Elaria"
    voice["style"] = elarion_prompt
    voice["chunk_size"] = 1200

    messages, audio_ids = prepare_generation_context(
        scene_prompt=voice["style"],
        ref_audio=None,
        ref_audio_in_system_message=True,
        audio_tokenizer=tokenizer,
        speaker_tags=[]
    )

    voice["temperature"] = None
    voice["top_k"] = None
    voice["top_p"] = None

    voice["messages"] = messages
    voice["audio_ids"] = audio_ids

    filename = "Voice.wav"
    sample = "The frost clung to the walls of Frosthaven"

    # The main text
    result, chunked_size_success = synthesization_loop(client_model,
                                                       sample,
                                                       voice,
                                                       filename)



def scenarios():

    logger.info(f"Starting Vocalis - Synthesizing Audio")
    # Load models
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

    input_file = "./input/scenarios/book.json"
    book = json.load(open(input_file))

    # only get the first voice for testing
    # voices = [voices[0], voices[3]]

    # check if the output folder exists, if not create it
    output_folder = "./output/scenarios"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for entry in book[:]:

        try:

            # convert the entry type to the PageType enum
            page_type = PageType[entry["type"]]

            match page_type:
                case PageType.TITLE:
                    logger.info(f"Title: {entry['title']}")
                    continue

                case PageType.SCENARIO:
                    # logger.info(f"Scenario: {entry['number']} {entry['title']}")

                    if entry["number"] != "68":
                        continue

                    # pad number to be 3 digits
                    number = str(entry['number']).zfill(3)

                    # Create a folder for the scenario
                    scenario_folder = os.path.join(output_folder, f"{number} - {entry['title']}")
                    if not os.path.exists(scenario_folder):
                        os.makedirs(scenario_folder)

                    sections_to_voice = list()

                    # Retrieve the sections to voice, all those that start with "Introduction"
                    for section in entry["sections"]:
                        if section["header"].startswith("Introduction"):
                            section_to_voice = {
                                "header": section["header"],
                                "text": section["text"]
                            }

                            # Add it to the list of sections to voice
                            sections_to_voice.append(section_to_voice)

                    # Write the manifest file
                    manifest_file = os.path.join(scenario_folder, "manifest.json")

                    # Check if the manifest file already exists, if it does, read it and update it with the new manifest, otherwise create a new manifest file
                    if os.path.exists(manifest_file):
                        with open(manifest_file, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                    else:
                        manifest = {
                            "scenario": {
                                "number": number,
                                "title": entry["title"]
                            },
                            "clips": []
                        }

                    clips: list[dict] = manifest["clips"]

                    # build manifest clips once (header + shared text)
                    for section in sections_to_voice:

                        # check if the header already exists in the manifest, if it does, skip it, otherwise add it to the manifest
                        if not any(clip["header"] == section["header"] for clip in clips):
                            manifest["clips"].append({
                                "header": section["header"],
                                "text": section["text"],
                                "audio": list()
                            })
                        else:
                            # make sure the text is still correct
                            for clip in manifest["clips"]:
                                if clip["header"] == section["header"]:
                                    clip["text"] = section["text"]


                    # Per voice, synthesize the audio for the sections to voice
                    for voice in voices:
                        # Creat the folder that is needed for the voice
                        voice_folder = os.path.join(scenario_folder, voice["name"])
                        if not os.path.exists(voice_folder):
                            os.makedirs(voice_folder)

                        # The title
                        # filename = "Title.wav"
                        # filename = os.path.join(voice_folder, filename)
                        print("there is a clip")
                        for clip in clips:

                            filename = f"{clip['header']}.wav"
                            filename = os.path.join(voice_folder, filename)

                            combined_text = "..." + entry["title"] + "\n...\n" + clip["text"]

                            # The main text
                            result, chunked_size_success = synthesization_loop(client_model,
                                                                               combined_text,
                                                                               voice,
                                                                               filename)

                            if result:
                                audio = {
                                    "voice": voice["name"],
                                    "file": f"{voice['name']}/{clip['header']}.wav",
                                    "chunk_size": chunked_size_success,
                                    "creation_time": datetime.datetime.now().isoformat()
                                }

                                # check if audio is already in the manifest, if it is, replace it
                                existing_audio = next((a for a in clip["audio"] if a["file"] == audio["file"]), None)

                                if existing_audio is not None:
                                    existing_audio.update(audio)
                                else:
                                    clip["audio"].append(audio)

                                logger.info(f"Chunk size found is : {chunked_size_success}")

                                with open(manifest_file, "w", encoding="utf-8") as f:
                                    json.dump(manifest, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(e)

if __name__ == "__main__":
    main()
