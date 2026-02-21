# Create the folders and synthesize the audio.
import json
import os
from enum import Enum, auto
from typing import Any

class PageType(Enum):
    TITLE = auto()
    SCENARIO = auto()
    CONTINUED_SCENARIO = auto()
    UNKNOWN = auto()

def synthesize_audio(text, voice_sample, filename):

    # touch the filename, so the file exists
    with open(filename, "w") as f:
        f.write("")


def main():

    voices = ["victor", "broom_salesman"]

    input_file = "./input/scenarios/book.json"
    book = json.load(open(input_file))

    # check if the output folder exists, if not create it
    output_folder = "./output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for entry in book:
        # convert the entry type to the PageType enum
        page_type = PageType[entry["type"]]

        match page_type:
            case PageType.TITLE:
                print(f"Title: {entry['title']}")

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
                    manifest["clips"].append({
                        "header": section["header"],
                        "text": section["text"],
                        "audio": {}
                    })

                # Per voice, synthesize the audio for the sections to voice
                for voice in voices:
                    # Creat the folder that is needed for the voice
                    voice_folder = os.path.join(scenario_folder, voice)
                    if not os.path.exists(voice_folder):
                        os.makedirs(voice_folder)

                    for clip in clips:
                        filename = f"{clip['header']}.wav"
                        filename = os.path.join(voice_folder, filename)

                        synthesize_audio(clip["text"], voice, filename)

                        clip["audio"][voice] = f"{voice}/{clip['header']}.wav"

                # Write the manifest file
                manifest_file = os.path.join(scenario_folder, "manifest.json")
                with open(manifest_file, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()