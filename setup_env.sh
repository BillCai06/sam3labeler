#!/bin/bash
# Environment setup for qwen3vl2sam
# Auto-detects NVIDIA (CUDA) or AMD (ROCm) GPU, falls back to CPU
#
# Usage:
#   bash setup_env.sh              # auto-detect
#   bash setup_env.sh --cuda       # force CUDA
#   bash setup_env.sh --rocm       # force ROCm (AMD)
#   bash setup_env.sh --cpu        # force CPU only

set -e

ENV_NAME="sam3labeler"
PYTHON="3.11"

# ── GPU backend detection ──────────────────────────────────────────────────────

detect_backend() {
    # Check for explicit override first
    for arg in "$@"; do
        case "$arg" in
            --cuda) echo "cuda"; return ;;
            --rocm) echo "rocm"; return ;;
            --cpu)  echo "cpu";  return ;;
        esac
    done

    # NVIDIA: nvidia-smi present and a GPU responds
    if command -v nvidia-smi &>/dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
        echo "cuda"
        return
    fi

    # AMD: rocminfo detects a GPU (also tries the DXG path used in WSL2)
    if command -v rocminfo &>/dev/null; then
        if HSA_ENABLE_DXG_DETECTION=1 rocminfo 2>/dev/null | grep -q "Vendor Name:.*AMD"; then
            echo "rocm"
            return
        fi
    fi

    echo "cpu"
}

BACKEND=$(detect_backend "$@")

# ── Per-backend config ─────────────────────────────────────────────────────────

case "$BACKEND" in
    cuda)
        # Detect installed CUDA version for the right wheel index
        if command -v nvcc &>/dev/null; then
            CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        else
            CUDA_VER="12.4"   # sensible default
        fi
        CUDA_TAG="cu$(echo $CUDA_VER | tr -d '.')"
        TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
        BACKEND_LABEL="CUDA ${CUDA_VER} (${CUDA_TAG})"
        ;;
    rocm)
        # Detect installed ROCm version (major.minor only, e.g. 7.2)
        if [ -f /opt/rocm/.info/version ]; then
            ROCM_VER=$(grep -oP '^\d+\.\d+' /opt/rocm/.info/version)
        elif [ -f /opt/rocm/include/rocm_version.h ]; then
            MAJOR=$(grep ROCM_VERSION_MAJOR /opt/rocm/include/rocm_version.h | awk '{print $3}')
            MINOR=$(grep ROCM_VERSION_MINOR /opt/rocm/include/rocm_version.h | awk '{print $3}')
            ROCM_VER="${MAJOR}.${MINOR}"
        fi
        ROCM_VER="${ROCM_VER:-7.2}"   # fallback
        TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VER}"
        BACKEND_LABEL="ROCm ${ROCM_VER}"
        ;;
    cpu)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        BACKEND_LABEL="CPU only"
        ;;
esac

# ── Setup ──────────────────────────────────────────────────────────────────────

echo "======================================"
echo "Setting up: $ENV_NAME"
echo "Backend:    $BACKEND_LABEL"
echo "PyTorch:    $TORCH_INDEX"
echo "======================================"

if [ "$BACKEND" = "rocm" ]; then
    # WSL2 AMD via DXG path (/dev/dxg instead of /dev/kfd)
    if [ ! -e /dev/kfd ] && [ ! -e /dev/dxg ]; then
        echo "WARNING: Neither /dev/kfd nor /dev/dxg found."
        echo "Install librocdxg: https://github.com/ROCm/librocdxg"
        echo "Continuing anyway..."
    fi
fi

conda create -n "$ENV_NAME" python="$PYTHON" -y
conda activate "$ENV_NAME" || source activate "$ENV_NAME"

# PyTorch (backend-specific wheel)
pip install torch torchvision --index-url "$TORCH_INDEX"

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
echo "Backend: $BACKEND_LABEL — verify GPU with:"
if [ "$BACKEND" = "cuda" ]; then
    echo "  python -c \"import torch; print(torch.cuda.get_device_name(0))\""
elif [ "$BACKEND" = "rocm" ]; then
    echo "  python -c \"import torch; print(torch.cuda.get_device_name(0))\""
    echo ""
    echo "NOTE: These env vars must be set for AMD (already in ~/.bashrc):"
    echo "  export HSA_ENABLE_DXG_DETECTION=1"
    echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
else
    echo "  python -c \"import torch; print(torch.__version__)\""
fi
echo ""
echo "Optional: 4-bit quantization"
echo "  pip install bitsandbytes"
echo ""
echo "Optional: SAM1 fallback"
echo "  pip install git+https://github.com/facebookresearch/segment-anything.git"
echo "======================================"

echo ""
echo "Done! Activate env with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Then run:"
echo "  python run_batch.py --gui                           # Launch Gradio GUI"
echo "  python run_batch.py --api                           # Launch FastAPI backend"
echo "  python run_batch.py -i ./images -c car person truck # CLI batch"
