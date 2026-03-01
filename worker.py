# Create the folders and synthesize the audio.
import json
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from enum import Enum, auto

import chunker
from boson import initialize_synthesization, sync_voice_prompts
from examples.generation import prepare_generation_context
import torch

import soundfile as sf

class PageType(Enum):
    TITLE = auto()
    SCENARIO = auto()
    CONTINUED_SCENARIO = auto()
    UNKNOWN = auto()

def synthesize_audio(model_client, tokenizer, text, voice_sample, filename):

    if voice_sample == "None":
        voice_sample = None

    chunked_text = chunker.chunk_text(text)

    style = voice_sample["style"]

    # style = (
    #     "You are a deep, sonorous narrator. "
    #     "Voice is cold, restrained, heavy with foreboding. "
    #     "Pause before key revelations. "
    #     "Emphasize important words strongly. "
    #     "Intensity rises on threats and stakes. "
    #     "Keep pacing deliberate."
    # )

    messages, audio_ids = prepare_generation_context(
        scene_prompt=style,
        ref_audio=voice_sample["name"],
        ref_audio_in_system_message=True,
        audio_tokenizer=tokenizer,
        speaker_tags=[],
    )

    # was .65
    temperature = 0.75
    top_k = 80
    top_p = 0.97
    ras_win_len = 4 + 20
    ras_win_max_num_repeat = 3 + 1
    generation_chunk_buffer_size = 2

    temperature = 0.75
    top_k = 70
    top_p = 0.95
    ras_win_len = 32
    ras_win_max_num_repeat = 4
    generation_chunk_buffer_size = 1

    temperature = voice_sample["temperature"]
    top_k = voice_sample["top_k"]
    top_p = voice_sample["top_p"]

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
        seed=1001,
    )

    sf.write(filename, concat_wv, sr)
    torch.cuda.empty_cache()

def main():

    # Prepare the model client and tokenizer for synthesization
    sync_voice_prompts()
    model_client, tokenizer = initialize_synthesization()

    voices = ["victor", "librarian", "sherlock", "slayer",
              "first", "en_man", "en_woman", "broom_salesman",
              "belinda", "morgan", "wick"]

    voices = ["victor", "first", "librarian"]

    voices = []

    style = (
        "Narrate with controlled intensity."
        "Maintain steady pacing."
        "When a sentence is short and declarative, slow slightly and add weight."
        "Pause briefly before final decisive statements. "
        "Avoid neutral tone on climactic lines."
    )

    voice = { "name": "fred",
              "temperature": 0.70,
              "top_k": 60,
              "top_p": 0.95,
              "style": style}

    voices.append(voice)

    style = (
        "Narrate with controlled intensity."
        "Maintain steady pacing."
        "When a sentence is short and declarative, slow slightly and add weight."
        "Pause briefly before final decisive statements. "
        "Avoid neutral tone on climactic lines."
    )

    voice = { "name": "victor",
              "temperature": 0.85,
              "top_k": 80,
              "top_p": 0.95,
              "style": style}

    # voices.append(voice)

    input_file = "./input/scenarios/book.json"
    book = json.load(open(input_file))

    # check if the output folder exists, if not create it
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for entry in book[1:]:

        try:

            # convert the entry type to the PageType enum
            page_type = PageType[entry["type"]]

            match page_type:
                case PageType.TITLE:
                    print(f"Title: {entry['title']}")
                    continue

                case PageType.SCENARIO:
                    print(f"Scenario: {entry['number']} {entry['title']}")

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
                                "audio": {}
                            })

                    # Per voice, synthesize the audio for the sections to voice
                    for voice in voices:
                        # Creat the folder that is needed for the voice
                        voice_folder = os.path.join(scenario_folder, voice["name"])
                        if not os.path.exists(voice_folder):
                            os.makedirs(voice_folder)

                        for clip in clips:
                            filename = f"{clip['header']}.wav"
                            filename = os.path.join(voice_folder, filename)

                            #synthesize_audio(model_client, tokenizer, clip["text"], voice, filename)

                            clip["audio"][voice["name"]] = f"{voice['name']}/{clip['header']}.wav"

                    with open(manifest_file, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()