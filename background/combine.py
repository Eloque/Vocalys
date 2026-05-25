import os
import json
import random

from torch.onnx._internal.fx import passes

from combiner import add_background_to_voice

# got through all the scenarios
#
scenario_folder = "./output/scenarios"

scenarios = [f for f in os.listdir(scenario_folder) if os.path.isdir(os.path.join(scenario_folder, f))]

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
#     # if manifest["scenario"]["number"] != "068":
#     #     continue
#
#     print(scenario)
#     print(manifest["scenario"]["number"])
#
#     # go through all the clips
#     for clip in manifest["clips"]:
#         for audio in clip["audio"]:
#             filename = os.path.join(folder, audio["file"])
#             # add_background_to_voice(filename)
#             add_background_to_voice(filename, background_file="./background/tracks/319_Shamans_Hollow.mp3", intro_delay_sec=12, voice_gain_db=0, bg_gain_db=-3)
#
# exit()

section_folder = "./output/sections"

sections = [f for f in os.listdir(section_folder) if os.path.isdir(os.path.join(section_folder, f))]

track_folder = "./background/tracks"

# load all the tracks in to a list
tracks = [f for f in os.listdir(track_folder) if f.endswith(".mp3")]

for section in sections:

    try:

        # combine to get the proper folder
        folder = os.path.join(section_folder, section)

        manifest = []

        # find the manifest files in the scenario folder
        manifest_files = [f for f in os.listdir(folder) if f.endswith(".json")]
        for manifest_file in manifest_files:
            manifest = json.load(open(os.path.join(folder, manifest_file)))

        print(section)
        print(manifest["section"]["number"])

        # go through all the clips
        for clip in manifest["section"]["audio"]:
            filename = os.path.join(folder, clip["file"])
            #add_background_to_voice(filename, background_file="./background/tracks/491_Red_Dragon_Dawn.mp3", intro_delay_sec=6, voice_gain_db=0, bg_gain_db=-5)

            # pick a random track
            track = tracks[random.randint(0, len(tracks) - 1)]
            track = tracks[random.randint(0, len(tracks) - 1)]
            print(track)

            # check if the file already exists, if it does, skip the synthesis
            if os.path.exists(filename[:-4] + "_background.opus"):
                print(f"File {filename} already exists, skipping synthesis.")
                continue

            add_background_to_voice(filename, f"./background/tracks/{track}", intro_delay_sec=3, voice_gain_db=0, bg_gain_db=-3)

    except Exception as e:
        print (e)
        pass
