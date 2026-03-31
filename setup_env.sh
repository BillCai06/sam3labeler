#!/bin/bash
# Environment setup for qwen3vl2sam
# Creates a conda env with all dependencies

set -e

ENV_NAME="qwen3vl2sam"
PYTHON="3.11"

echo "======================================"
echo "Setting up: $ENV_NAME"
echo "======================================"

# Create conda env
conda create -n $ENV_NAME python=$PYTHON -y
conda activate $ENV_NAME || source activate $ENV_NAME

# PyTorch with CUDA 12.4 (adjust index-url for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install \
    transformers>=4.45.0 \
    accelerate \
    qwen-vl-utils \
    Pillow>=10.0.0 \
    opencv-python-headless \
    numpy>=1.24.0 \
    scipy \
    pyyaml \
    tqdm \
    aiofiles

# COCO tools
pip install pycocotools

# API
pip install "fastapi>=0.111.0" "uvicorn[standard]" python-multipart

# GUI
pip install "gradio>=4.36.0"

echo ""
echo "======================================"
echo "Installing SAM2 (from Meta GitHub)..."
echo "======================================"
pip install git+https://github.com/facebookresearch/sam2.git

echo ""
echo "======================================"
echo "Optional: 4-bit quantization (if you have a small GPU < 20GB VRAM)"
echo "  pip install bitsandbytes"
echo ""
echo "Optional: SAM1 fallback"
echo "  pip install git+https://github.com/facebookresearch/segment-anything.git"
echo "  # Download SAM1 checkpoint:"
echo "  mkdir -p checkpoints"
echo "  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
echo "======================================"

echo ""
echo "Done! Activate env with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Then run:"
echo "  python run_batch.py --gui                          # Launch Gradio GUI"
echo "  python run_batch.py --api                          # Launch FastAPI backend"
echo "  python run_batch.py -i ./images -c car person truck  # CLI batch"
