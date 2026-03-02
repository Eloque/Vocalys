from pathlib import Path
import json
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory

APP_ROOT = Path(__file__).resolve().parent
SCENARIO_ROOT = (APP_ROOT / ".." / "output").resolve()
app = Flask(__name__)

def find_scenario_folder_by_number(number: str) -> Path | None:
	number = number.strip()
	
	for folder in SCENARIO_ROOT.iterdir():
		if folder.is_dir() and folder.name.startswith(f"{number} - "):
			return folder
	
	return None

def load_manifest(folder: Path) -> dict:
	manifest_path = folder / "manifest.json"

	if not manifest_path.exists():
		raise FileNotFoundError(manifest_path)

	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

	return manifest


@app.route("/", methods=["GET", "POST"])
def index():

	if request.method == "POST":
		number = request.form.get("number", "").strip()

		if not number:
			return render_template("index.html", error="Enter a scenario number.")

		return redirect(url_for("scenario", number=number))

	scenarios = []

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

	# Sort numerically
	scenarios.sort(key=lambda s: int(s["number"]))

	return render_template("index.html", scenarios=scenarios)

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

	# Add: clip.audio_urls[voice] = URL to play that file
	for clip in manifest.get("clips", []):
		audio = clip.get("audio", {})
		audio_urls = {}
		if isinstance(audio, dict):
			for voice, relpath in audio.items():
				if isinstance(relpath, str) and relpath.strip():
					audio_urls[voice] = url_for("scenario_file", number=number, filename=relpath)
		clip["audio_urls"] = audio_urls

	return render_template("scenario.html", folder_name=folder.name, manifest=manifest)


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=5000, debug=True)