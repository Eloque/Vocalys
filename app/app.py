from pathlib import Path
import json
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory

APP_ROOT = Path(__file__).resolve().parent
SCENARIO_ROOT = (APP_ROOT / ".." / "output/scenarios").resolve()
SECTION_ROOT = (APP_ROOT / ".." / "output/sections").resolve()
app = Flask(__name__)

def find_scenario_folder_by_number(number: str) -> Path | None:
	number = number.strip()

	for folder in SCENARIO_ROOT.iterdir():
		if folder.is_dir() and folder.name.startswith(f"{number} - "):
			return folder

	return None


def find_section_folder_by_number(number: str) -> Path | None:
	number = number.strip()

	for folder in SECTION_ROOT.iterdir():
		if folder.is_dir() and folder.name == number:
			return folder

	return None

def load_manifest(folder: Path) -> dict:
	manifest_path = folder / "manifest.json"

	if not manifest_path.exists():
		raise FileNotFoundError(manifest_path)

	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

	return manifest


@app.route("/section/<number>/file/<path:filename>")
def section_file(number, filename):
	folder = find_section_folder_by_number(number)
	if folder is None:
		abort(404)

	return send_from_directory(folder, filename)
@app.route("/", methods=["GET", "POST"])
def index():

	if request.method == "POST":
		number = request.form.get("number", "").strip()

		if not number:
			return render_template("index.html", error="Enter a scenario number.")

		return redirect(url_for("scenario", number=number))

	scenarios = []
	sections = []

	for folder in SCENARIO_ROOT.iterdir():
		if not folder.is_dir():
			continue

		name = folder.name

		if " - " not in name:
			continue

		number_part, title_part = name.split(" - ", 1)

		if not number_part.isdigit():
			continue

		scenarios.append({
			"number": number_part,
			"title": title_part,
			"display": f"{number_part} - {title_part}"
		})

	for folder in SECTION_ROOT.iterdir():
		if not folder.is_dir():
			continue

		number_part = folder.name

		sections.append({
			"number": number_part,
			"display": f"{number_part}"
		})


	# Sort numerically
	scenarios.sort(key=lambda s: int(s["number"]))
	sections.sort(key=lambda s: str(s["number"]))

	return render_template("index.html", scenarios=scenarios, sections=sections)

@app.route("/s/<number>/file/<path:filename>")
def scenario_file(number: str, filename: str):
	folder = find_scenario_folder_by_number(number)
	if folder is None:
		abort(404)
	return send_from_directory(folder, filename, as_attachment=False)

from flask import url_for

@app.route("/s/<number>")
def scenario(number: str):
	folder = find_scenario_folder_by_number(number)
	if folder is None:
		abort(404)

	manifest = load_manifest(folder)
	scenario = manifest.get("scenario", {})

	# build one "clip"
	clips = [{
		"header": scenario.get("title", "Scenario"),
		"audio_urls": {}
	}]

	for item in scenario.get("audio", []):
		voice = item.get("voice")
		relpath = item.get("file")

		# relpath replace .wav with _background.opus
		relpath = relpath.replace(".wav", "_background.opus")

		if voice and relpath:
			clips[0]["audio_urls"][voice] = url_for(
				"scenario_file",
				number=number,
				filename=relpath
			)

	# inject into manifest so template stays identical
	manifest["clips"] = clips

	return render_template(
		"scenario.html",
		folder_name=folder.name,
		manifest=manifest
	)

	# folder = find_scenario_folder_by_number(number)
	# if folder is None:
	# 	abort(404)
	#
	# manifest = load_manifest(folder)
	#
	# # --- build title clip ---
	# scenario = manifest.get("scenario", {})
	# title_clip = {
	# 	"header": scenario.get("title", "Title"),
	# 	"audio_urls": {}
	# }
	#
	# for audio in scenario.get("audio", []):
	# 	voice = audio.get("voice")
	# 	relpath = audio.get("file")
	#
	# 	if voice and relpath:
	# 		title_clip["audio_urls"][voice] = url_for(
	# 			"scenario_file",
	# 			number=number,
	# 			filename=relpath
	# 		)
	#
	# # --- build normal clips ---
	# for audio in scenario.get("audio", []):
	# 	audio_urls = {}
	#
	# 	for item in audio.get("audio", []):
	# 		voice = item.get("voice")
	# 		relpath = item.get("file")
	#
	# 		if voice and relpath:
	# 			audio_urls[voice] = url_for(
	# 				"scenario_file",
	# 				number=number,
	# 				filename=relpath
	# 			)
	#
	# 	audio["audio_urls"] = audio_urls
	#
	# # --- prepend title ---
	# # manifest["clips"].insert(0, title_clip)
	#
	# return render_template(
	# 	"scenario.html",
	# 	folder_name=folder.name,
	# 	manifest=manifest
	# )

@app.route("/section/<number>")
def section(number: str):
	folder = find_section_folder_by_number(number)
	if folder is None:
		abort(404)

	manifest = load_manifest(folder)
	section = manifest.get("section", {})

	# build one "clip"
	clips = [{
		"header": section.get("number", "Section"),
		"audio_urls": {}
	}]

	for item in section.get("audio", []):
		voice = item.get("voice")
		relpath = item.get("file")

		# relpath replace .wav with _background.opus
		relpath = relpath.replace(".wav", "_background.opus")

		if voice and relpath:
			clips[0]["audio_urls"][voice] = url_for(
				"section_file",
				number=number,
				filename=relpath
			)

	# inject into manifest so template stays identical
	manifest["clips"] = clips

	return render_template(
		"section.html",
		folder_name=folder.name,
		manifest=manifest
	)

if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)