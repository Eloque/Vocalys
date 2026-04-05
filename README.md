# 🎙️ Vocalys

> **Vocalys — Local hosted TTS pipeline for long-form storytelling and narration.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?logo=flask)
![AI](https://img.shields.io/badge/AI-TTS-green)
![Audio](https://img.shields.io/badge/Audio-Processing-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow?logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-TTS-red?logo=pytorch)

---
🚧 **This project is very much a work in progress.**  
Expect rough edges, breaking changes, and incomplete features.
Expect the author to mess around, try things, get better, get worse and then in the end produce something worthwhile. Perhaps. 

## 📚 What is Vocalys?
Vocalys is an experimental, local-first AI narration tool designed for **Frosthaven** narration.

It reads the text data from the PDF files, then it  utilizes **Higgs Boson AI** and specifically **Faster Higgs Audio** to synthesize the audio.

A simple Flask app is provided then to select and playback scenario's and sections.

No cloud required. Everything runs locally. You do need to generate the audio however. On a 3080, at 8 bts quantized, this takes serious time, do not expect realtime. It is possible to run on CPU. That takes about 3000 seconds, 50 minutes, for scarcely 2 minute of audio. That is on a 5900X 12 Core with 64 GB of memory

---
## ✨ Features (so far)

- 📄 **PDF Reader & Analyzer**  
  Extracts and processes the data found in the Forsthaven PDFs

- 🔊 **Audio Synthesizer**  
  Generate audio based on voice profiles, cloning existing clips.

- 🌐 **Flask Web Interface**  
  Simple UI to interact with the resulted audio

---
## 🧩 Components
Vocalys is built as a modular PDF - TTS pipeline:

### 1. PDF Analyzer
- Parses PDF documents
- Extracts sections and text to JSON

### 2. Audio Synthesizer
- Converts text into speech
- Supports voice cloning / voice samples
- Optimized for long-form narration

### 3. Flask Web App
- Minimal interface
- Trigger processing and playback
- Designed for local usage

---

## 🧠 How it works

1. Follow the instruction to clone the parts of Faster-Higgs and Worldhaven that are needed
2. Run PDF Background Stripper
3. Run scenarios.py and sections.py to extract the PDF's to JSON
4. Supply and describe voices, in voices.json
5. Run worker. Wait a very long time. Though it does resume when interrupted and started again.
6. Run the Flask App.

---
## 🎤 Voice & Audio
- Audio is **not pre-generated**
- Users must generate audio themselves
- Supports using **custom voice samples**

This allows for flexible and personalized narration.
---
## 🖥️ Local-first

- Runs entirely on your machine
- No external APIs required
- Designed for privacy and control and cost saving
---
## 🚀 Quick Start

```
# Clone repo
git clone https://github.com/yourname/vocalys.git
cd vocalys

# Install external repos, don't pull everyting, that is huge!
git clone --filter=blob:none --no-checkout https://github.com/any2cards/worldhaven.git
cd worldhaven
git sparse-checkout init --cone
git sparse-checkout set images/books/frosthaven
git checkout
cd ..

git submodule init
git submodule update --recursive faster-higgs-audio/

# Create venv and Install dependencies
uv venv
source .venv/bin/activate
uv pip install -r faster-higgs-audio/requirements.txt -e faster-higgs-audio bitsandbytes
uv pip install pypdf
uv pip install pdfplumber
uv pip install scikit-image
```

Currently, it is also required to pull specific Tokenizer and Model from Huggingface.
The latest updates from Higgs broke some things that have not been resolved yet.

You need to have Hugging Face Hub installed for this
```
uv pip install huggingface-hub
mkdir models
cd models

huggingface-cli download bosonai/higgs-audio-v2-tokenizer --revision 9d4988fbd4ad07b4cac3a5fa462741a41810dbec --local-dir ./tokenizer_old --local-dir-use-symlinks False
huggingface-cli download bosonai/higgs-audio-v2-generation-3B-base --revision 10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84 --local-dir ./model_old --local-dir-use-symlinks False
```
---

## 🧪 Intended Use
- Frosthaven narration
- Experimentation with local TTS pipelines

---
## ⚠️ Disclaimer
This is a personal, experimental project.

- Not production ready
- Not optimized
- Not for profit, only fun.
---
## 📜 License
MIT License

---
## 💡 Notes

This project is part exploration, part tool — built to learn, experiment, and push local AI narration forward. It's main purpose is to provide me with an outlet for my programming hobbies.

If you try it: expect some fiddling, maybe chaos, but hopefully useful chaos 😄
