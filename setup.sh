#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# 🎙️ Vocalys Setup Script
# ─────────────────────────────────────────────

echo "🎙️ Setting up Vocalys..."

# ─────────────────────────────────────────────
# 🔍 Check dependencies
# ─────────────────────────────────────────────

command -v git >/dev/null || { echo "❌ git not found"; exit 1; }
command -v uv >/dev/null || { echo "❌ uv not found (pip install uv)"; exit 1; }

# ─────────────────────────────────────────────
# 📁 Clone worldhaven (sparse)
# ─────────────────────────────────────────────

if [ ! -d "worldhaven" ]; then
echo "📦 Cloning worldhaven (sparse)..."
git clone --filter=blob:none --no-checkout https://github.com/any2cards/worldhaven.git
cd worldhaven
git sparse-checkout init --cone
git sparse-checkout set images/books/frosthaven
git checkout
cd ..
else
echo "✅ worldhaven already present"
fi

# ─────────────────────────────────────────────
# 🔗 Submodules
# ─────────────────────────────────────────────

echo "🔗 Initializing submodules..."
git submodule update --init --recursive faster-higgs-audio/

# ─────────────────────────────────────────────
# 🐍 Python environment
# ─────────────────────────────────────────────

if [ ! -d ".venv" ]; then
echo "🐍 Creating virtual environment..."
uv venv --python 3.10
else
echo "✅ Virtual environment already exists"
fi

source .venv/bin/activate

echo "📦 Installing Python dependencies..."

uv pip install -r faster-higgs-audio/requirements.txt -e faster-higgs-audio bitsandbytes
uv pip install pypdf pdfplumber scikit-image huggingface-hub patches

# ─────────────────────────────────────────────
# 🤗 Models
# ─────────────────────────────────────────────

mkdir -p models
cd models

hf_download() {
	local repo="$1"
	local revision="$2"
	local target_dir="$3"

	hf download "$repo" \
		--revision "$revision" \
		--local-dir "$target_dir" \
		--local-dir-use-symlinks False
}

download_if_missing() {
	local target_dir="$1"
	local repo="$2"
	local revision="$3"

	if [ -d "$target_dir" ]; then
		printf "[OK] %s already exists, skipping\n" "$target_dir"
		return 0
	fi

	printf "[DL] %s -> %s\n" "$repo" "$target_dir"

	if ! hf_download "$repo" "$revision" "$target_dir"; then
		printf "[ERR] Failed to download %s\n" "$repo" >&2
		return 1
	fi

	printf "[OK] Download complete: %s\n" "$target_dir"
}

download_if_missing "tokenizer_old" "bosonai/higgs-audio-v2-tokenizer" "9d4988fbd4ad07b4cac3a5fa462741a41810dbec"
download_if_missing "model_old" "bosonai/higgs-audio-v2-generation-3B-base" "10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84"

cd ..

# ─────────────────────────────────────────────
# ✅ Done
# ─────────────────────────────────────────────

echo ""
echo "✅ Vocalys setup complete!"
echo ""
echo "👉 Activate environment:"
echo "   source .venv/bin/activate"
echo ""
echo "👉 Run app (example):"
echo "   python app.py"
echo ""
