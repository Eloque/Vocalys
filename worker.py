# Create the folders and synthesize the audio.
import datetime
import json
import os
import time

from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

from enum import Enum, auto

import chunker
from boson import initialize_synthesization, sync_voice_prompts, get_device
from examples.generation import prepare_generation_context, HiggsAudioModelClient
import torch

import soundfile as sf

class PageType(Enum):
    TITLE = auto()
    SCENARIO = auto()
    CONTINUED_SCENARIO = auto()
    UNKNOWN = auto()

def synthesize_audio(model_client, tokenizer, text, voice_sample, filename, max_chunk_size=120):

    # check if the files already exists, if it does, skip the synthesis
    if os.path.exists(filename):
        # print(f"File {filename} already exists, skipping synthesis.")
        return

    if voice_sample == "None":
        voice_sample = None

    print(f"File {filename} does not exist, missing {voice_sample['name']} voice sample, synthesizing...")

    device = get_device("auto")
    device_id = None if device == "cpu" else int(device.split(":")[-1])

    # Chunk the text into smaller pieces based on the max_chunk_size
    chunked_text = chunker.chunk_text(text, max_chunk_size)
    primed  = False

    # only keep the first 1 chunks for testing
    # chunked_text = chunked_text[:1]
    # check if voice has a primer, if it does, add it to the beginning of the chunked text
    if "primer" in voice_sample and voice_sample["primer"] is not None:
        chunked_text.insert(0, voice_sample["primer"])
        primed = True

    style = voice_sample["style"]

    messages, audio_ids = prepare_generation_context(
        scene_prompt=style,
        ref_audio=voice_sample["name"],
        ref_audio_in_system_message=True,
        audio_tokenizer=tokenizer,
        speaker_tags=[],
    )

    print(f"{voice_sample['name']} context generated.")

    ras_win_len = 32
    ras_win_max_num_repeat = 4
    generation_chunk_buffer_size = 3

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
        seed=1001001,
        primed=primed
    )

    sf.write(filename, concat_wv, sr)
    torch.cuda.empty_cache()

def main():

    # Load voices
    voices = json.load(open("./voices/voices.json"))
    voices = voices["voices"]

    sync_voice_prompts()

    input_file = "./input/scenarios/book.json"
    book = json.load(open(input_file))

    # only get the first voice for testing
    # voices = [voices[0], voices[3]]

    # check if the output folder exists, if not create it
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # # Load the tokenizer
    # tokenizer = load_higgs_audio_tokenizer("bosonai/higgs-audio-v2-tokenizer", device=get_device("cpu"))
    #
    # model_client = HiggsAudioModelClient(
    #     model_path="bosonai/higgs-audio-v2-generation-3B-base",
    #     audio_tokenizer=tokenizer,
    #     device_id=device_id,
    #     max_new_tokens=4096, # 378, # $4096 / 8,
    #     use_static_kv_cache=False,
    #     use_quantization=True,
    #     quantization_bits=8,
    # )

    client_model, tokenizer = initialize_synthesization()

    for entry in book[:]:

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

                    if entry['number'] != '62':
                        continue

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

                    # Per voice, synthesize the audio for the sections to voice
                    for voice in voices:
                        # Creat the folder that is needed for the voice
                        voice_folder = os.path.join(scenario_folder, voice["name"])
                        if not os.path.exists(voice_folder):
                            os.makedirs(voice_folder)

                        for clip in clips:
                            filename = f"{clip['header']}.wav"
                            filename = os.path.join(voice_folder, filename)

                            synthesizing = True
                            max_chunk_size = 600

                            # check if voice has a chunk size, if it does, use it as the initial chunk size for synthesis, otherwise use the default chunk size
                            if "chunk_size" in voice and voice["chunk_size"] is not None:
                                max_chunk_size = voice["chunk_size"]
                            else:
                                voice["chunk_size"] = max_chunk_size

                            # check if the files already exists, if it does, skip the synthesis
                            if os.path.exists(filename):
                                print(f"File {filename} already exists, skipping synthesis.")
                                continue

                            while synthesizing:
                                oom = False

                                try:
                                    start = time.perf_counter()
                                    print(f"Synthesizing with chunk size: {max_chunk_size}")
                                    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)

                                    # synthesize_audio(client_model, tokenizer, clip["text"], voice, filename, max_chunk_size=max_chunk_size)
                                    synthesize_audio(client_model, tokenizer, clip["text"], voice, filename,
                                                     max_chunk_size=max_chunk_size)

                                    elapsed = time.perf_counter() - start
                                    print(f"{elapsed:.2f}s to synthesize {filename}")

                                    synthesizing = False

                                except torch.OutOfMemoryError:
                                    print("OOM on clip:", clip["header"])
                                    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)
                                    oom = True

                                if oom:
                                    print("Leaving exception handler")
                                    print(torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.memory_reserved() / 1024 ** 2)

                                    if max_chunk_size > 100:
                                        max_chunk_size -= 20
                                        print(f"Retrying with smaller chunk size: {max_chunk_size}")
                                    else:
                                        print(f"Error synthesizing audio for {filename}: OOM, no chunk size left, skipping.")
                                        synthesizing = False

                            audio = {
                                "voice": voice["name"],
                                "file": f"{voice['name']}/{clip['header']}.wav",
                                "chunk_size": max_chunk_size,
                                "creation_time": datetime.datetime.now().isoformat()
                            }

                            # check if audio is already in the manifest, if it is, replace it
                            existing_audio = next((a for a in clip["audio"] if a["file"] == audio["file"]), None)

                            if existing_audio is not None:
                                existing_audio.update(audio)
                            else:
                                clip["audio"].append(audio)

                            print(f"Chunk size found is : {max_chunk_size}")
                            
                            with open(manifest_file, "w", encoding="utf-8") as f:
                                json.dump(manifest, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()