import os
import json
from combiner import add_background_to_voice


# got through all the scenarios
#
# scenario_folder = "./output/scenarios"
#
# scenarios = [f for f in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, f))]
#
# for scenario in scenarios:
#
#     # combine to get the proper folder
#     folder = os.path.join(scenario_folder, scenario)
#
#     manifest = []
#
#     # find the manifest files in the scenario folder
#     manifest_files = [f for f in os.listdir(folder) if f.endswith(".json")]
#     for manifest_file in manifest_files:
#         manifest = json.load(open(os.path.join(folder, manifest_file)))
#
#     if manifest["scenario"]["number"] != "068":
#         continue
#
#     print(scenario)
#     print(manifest["scenario"]["number"])
#
#     # go through all the clips
#     for clip in manifest["clips"]:
#         for audio in clip["audio"]:
#
#             filename = os.path.join(folder, audio["file"])
#             add_background_to_voice(filename)

section_folder = "./output/sections"

sections = [f for f in os.listdir(section_folder) if os.path.isdir(os.path.join(section_folder, f))]

for section in sections:

    if section != "106.4":
        continue

    # combine to get the proper folder
    folder = os.path.join(section_folder, section)

    manifest = []

    # find the manifest files in the scenario folder
    manifest_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    for manifest_file in manifest_files:
        manifest = json.load(open(os.path.join(folder, manifest_file)))

    # go through all the clips
    for clip in manifest["section"]["audio"]:
        filename = os.path.join(folder, clip["file"])
        add_background_to_voice(filename, background_file="./background/tracks/491_Red_Dragon_Dawn.mp3", intro_delay_sec=6, voice_gain_db=0, bg_gain_db=-5)