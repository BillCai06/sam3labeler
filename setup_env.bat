@echo off
REM ==========================================================
REM  qwen3vl2sam — Windows one-click setup
REM  Requirements: Anaconda/Miniconda + NVIDIA GPU (CUDA 12.4)
REM  Run from Anaconda Prompt (NOT PowerShell or cmd directly)
REM ==========================================================

setlocal enabledelayedexpansion

set ENV_NAME=qwen3vl2sam
set PYTHON_VER=3.11

echo ======================================
echo  Setting up: %ENV_NAME%
echo ======================================

REM Create conda environment
call conda create -n %ENV_NAME% python=%PYTHON_VER% -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment. Is Anaconda/Miniconda installed?
    pause & exit /b 1
)

REM Activate environment
call conda activate %ENV_NAME%
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment.
    pause & exit /b 1
)

REM PyTorch with CUDA 12.4
REM Change cu124 to cu118 / cu121 if your CUDA version differs.
REM Check your CUDA version with: nvidia-smi
echo.
echo Installing PyTorch (CUDA 12.4)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo [ERROR] PyTorch install failed. Check your internet connection or CUDA version.
    pause & exit /b 1
)

REM Core dependencies
echo.
echo Installing core dependencies...
pip install ^
    "transformers>=4.45.0" ^
    accelerate ^
    qwen-vl-utils ^
    "Pillow>=10.0.0" ^
    opencv-python ^
    "numpy>=1.24.0" ^
    scipy ^
    pyyaml ^
    tqdm ^
    aiofiles

REM COCO tools
pip install pycocotools-windows

REM API
pip install "fastapi>=0.111.0" "uvicorn[standard]" python-multipart

REM GUI
pip install "gradio>=4.36.0"

echo.
echo ======================================
echo  SAM3 is bundled locally in ./sam3/
echo  No extra install needed.
echo ======================================

echo.
echo ======================================
echo  Optional extras:
echo   4-bit quantization (GPU ^< 20 GB VRAM):
echo     pip install bitsandbytes
echo.
echo   SAM2 backend (requires nightly PyTorch for Blackwell GPUs):
echo     pip install git+https://github.com/facebookresearch/sam2.git
echo.
echo   SAM1 fallback:
echo     pip install git+https://github.com/facebookresearch/segment-anything.git
echo ======================================

echo.
echo Done! To use the environment:
echo   conda activate %ENV_NAME%
echo.
echo Then run:
echo   python run_batch.py --gui                         ^<-- Gradio GUI
echo   python run_batch.py --api                         ^<-- FastAPI backend
echo   python run_batch.py -i .\images -c car person     ^<-- CLI batch
echo.

pause
endlocal
