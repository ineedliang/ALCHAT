@echo off
setlocal EnableDelayedExpansion
title AI Chat - Environment Setup

echo.
echo ============================================================
echo   AI Chat - Automated Environment Setup
echo ============================================================
echo.

:: ---- Check Python ----
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found in PATH.
    echo  Please install Python 3.10 or newer from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%

:: ---- Check pip ----
echo [2/7] Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  pip not found, installing...
    python -m ensurepip --upgrade
)
echo  pip OK

:: ---- Upgrade pip ----
echo [3/7] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo  pip upgraded

:: ---- Check CUDA / GPU ----
echo [4/7] Detecting GPU...
python -c "import subprocess; r=subprocess.run(['nvidia-smi'], capture_output=True); exit(0 if r.returncode==0 else 1)" >nul 2>&1
if errorlevel 1 (
    echo  No NVIDIA GPU detected - will install CPU-only PyTorch.
    set USE_GPU=0
) else (
    echo  NVIDIA GPU detected.
    :: Detect CUDA version from nvidia-smi
    for /f "tokens=9 delims= " %%c in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%c
    echo  CUDA Version: !CUDA_VER!
    set USE_GPU=1
)

:: ---- Install PyTorch ----
echo [5/7] Installing PyTorch...
echo  (This may take several minutes - PyTorch is large)
echo.

if "!USE_GPU!"=="1" (
    echo  Installing PyTorch with CUDA 12.6 support...
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --quiet
    if errorlevel 1 (
        echo  CUDA 12.6 install failed, trying CUDA 12.4...
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --quiet
        if errorlevel 1 (
            echo  CUDA install failed, falling back to CPU PyTorch...
            python -m pip install torch torchvision torchaudio --quiet
        )
    )
) else (
    python -m pip install torch torchvision torchaudio --quiet
)

:: Verify torch installed
python -c "import torch; print('  PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo  ERROR: PyTorch installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

:: ---- Install PyAudio (Windows needs special handling) ----
echo [6/7] Installing PyAudio...
python -m pip install pyaudio --quiet 2>nul
if errorlevel 1 (
    echo  Standard pyaudio install failed, trying pipwin...
    python -m pip install pipwin --quiet
    python -m pipwin install pyaudio --quiet 2>nul
    if errorlevel 1 (
        echo.
        echo  WARNING: PyAudio could not be installed automatically.
        echo  Mic input will not work until PyAudio is installed.
        echo  Manual fix options:
        echo    Option A: pip install pipwin  then  pipwin install pyaudio
        echo    Option B: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        echo              then: pip install PyAudio-0.2.xx-cpXX-cpXX-win_amd64.whl
        echo.
    )
) else (
    echo  PyAudio installed OK
)

:: ---- Install remaining requirements ----
echo [7/7] Installing remaining dependencies...
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo.
    echo  WARNING: Some packages may have failed. Check output above.
    echo  The app may still run if core packages installed correctly.
    echo.
) else (
    echo  All dependencies installed successfully.
)

:: ---- Verify critical imports ----
echo.
echo ============================================================
echo   Verifying installation...
echo ============================================================
echo.

python -c "
packages = [
    ('torch',           'PyTorch'),
    ('transformers',    'Transformers'),
    ('PyQt6',           'PyQt6'),
    ('bitsandbytes',    'BitsAndBytes (4-bit GPU)'),
    ('accelerate',      'Accelerate'),
    ('pyttsx3',         'TTS'),
    ('speech_recognition', 'Speech Recognition'),
    ('requests',        'Requests'),
    ('bs4',             'BeautifulSoup'),
]
failed = []
for mod, name in packages:
    try:
        __import__(mod)
        print(f'  [OK]  {name}')
    except ImportError:
        print(f'  [!!]  {name} - MISSING')
        failed.append(name)

try:
    import pyaudio
    print('  [OK]  PyAudio (mic input)')
except:
    print('  [--]  PyAudio - not installed (mic input disabled)')

try:
    import obswebsocket
    print('  [OK]  OBS WebSocket')
except:
    print('  [--]  OBS WebSocket - not installed (OBS disabled)')

print()
if failed:
    print(f'  {len(failed)} package(s) missing: {chr(44).join(failed)}')
else:
    print('  All core packages verified.')
"

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo  To start the app, run:
echo.
echo     python main.py
echo.
echo  Or double-click: run.bat
echo.
echo  First run will download the default model (~1 GB).
echo  Models are cached in: %USERPROFILE%\ai_models
echo  Conversations saved in: %USERPROFILE%\ai_chat_saves
echo.
pause
