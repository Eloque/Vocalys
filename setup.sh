#!/bin/bash

# =============================================================================
# VOCALYS - Setup Script
# =============================================================================
# This script sets up the Vocalys project environment including:
# - Repository cloning and sparse checkouts
# - Submodule initialization
# - Virtual environment creation with UV
# - Dependency installation
# - Model downloads from Hugging Face
# =============================================================================

set -e  # Exit on error

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly BOLD='\033[1m'
readonly NC='\033[0m' # No Color

# Fancy symbols
readonly CHECK="${GREEN}✓${NC}"
readonly CROSS="${RED}✗${NC}"
readonly ARROW="${BLUE}→${NC}"
readonly STAR="${YELLOW}★${NC}"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BOLD}${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${MAGENTA}║${NC}  ${CYAN}${BOLD}$1${NC}${MAGENTA}${BOLD}"
    echo -e "${BOLD}${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${BOLD}${BLUE}▶${NC} ${BOLD}$1${NC}"
}

print_success() {
    echo -e "  ${CHECK} $1"
}

print_error() {
    echo -e "  ${CROSS} ${RED}$1${NC}"
}

print_info() {
    echo -e "  ${ARROW} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠${NC}  ${YELLOW}$1${NC}"
}

# =============================================================================
# Main Setup
# =============================================================================

print_header "VOCALYS SETUP"
echo -e "${CYAN}Welcome to the Vocalys setup script!${NC}\n"

# -----------------------------------------------------------------------------
# Step 1: Clone Main Repository
# -----------------------------------------------------------------------------
print_step "Cloning Vocalys repository..."

# Install external repos, don't pull everything, that is huge!
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

uv pip install huggingface-hub
mkdir models
cd models

huggingface-cli download bosonai/higgs-audio-v2-tokenizer --revision 9d4988fbd4ad07b4cac3a5fa462741a41810dbec --local-dir ./tokenizer_old --local-dir-use-symlinks False
huggingface-cli download bosonai/higgs-audio-v2-generation-3B-base \
    --revision 10840182ca4ad5d9d9113b60b9bb3c1ef1ba3f84 \
    --local-dir ./model_old \
    --local-dir-use-symlinks False
print_success "Generation model downloaded"

cd ..

# =============================================================================
# Setup Complete
# =============================================================================

print_header "SETUP COMPLETE!"

echo -e "${GREEN}${BOLD}All done! Your Vocalys environment is ready to use.${NC}\n"
echo -e "${CYAN}Next steps:${NC}"
echo -e "  ${ARROW} Activate the environment: ${BOLD}source .venv/bin/activate${NC}"
echo -e "  ${ARROW} Run the application as needed\n"

echo -e "${MAGENTA}${STAR}${NC} ${BOLD}Happy voice synthesizing!${NC} ${MAGENTA}${STAR}${NC}\n"