@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM  darask-paint AI Diffusion plugin launcher
REM  Starts a local ComfyUI instance + darask_server.py (this
REM  fork's headless HTTP adapter) for darask-paint's
REM  "AI seisei (Diffusion)" / "AI chikan (Diffusion)" menus.
REM
REM  - First run: installs uv, creates a Python env, installs
REM    PyTorch (CUDA 12.8 if an NVIDIA GPU is present, else CPU),
REM    downloads ComfyUI (comfyanonymous/ComfyUI, version pinned
REM    to match ai_diffusion/backend/resources.py) and its
REM    requirements, and (after asking) a default SD1.5 checkpoint.
REM    Everything is installed under %LOCALAPPDATA%\DaraskAIDiffusion
REM    so it does not touch any other ComfyUI/Krita installation.
REM    Total download size the first time: roughly 5-8GB (PyTorch with
REM    CUDA is ~3GB by itself, plus ComfyUI's own dependencies and the
REM    ~2GB default checkpoint). CPU-only PyTorch is smaller (~1GB) but
REM    generation will be much slower.
REM  - Later runs: starts ComfyUI + the API server immediately. Setup
REM    completion is tracked via a marker file (%LOCALAPPDATA%\DaraskAIDiffusion\.darask_setup)
REM    recording the exact ComfyUI commit, PyTorch/torchvision versions,
REM    and a hash of ComfyUI's requirements.txt -- not just "does the
REM    folder exist" -- so a version bump in this script triggers a
REM    clean re-setup instead of silently running a stale environment.
REM  - API server: http://127.0.0.1:8424 (local only; darask-paint
REM    talks to /api/v1/health, /api/v1/generate, /api/v1/inpaint).
REM  - Close this window (or Ctrl+C) to stop both ComfyUI and the
REM    API server.
REM  - Note: cloning this repository does NOT require git submodules.
REM    darask_server.py does not import the ai_diffusion package (the
REM    Krita plugin code) at all, so ai_diffusion/websockets and
REM    ai_diffusion/debugpy are unused here (they only matter if you
REM    also use this fork as a Krita plugin).
REM ============================================================

set "COMFY_REPO_URL=https://github.com/comfyanonymous/ComfyUI"
set "COMFY_VERSION=a95e461916de9cbda2e89140ab86a8a7c3f9702a"
set "TORCH_VERSION=2.11.0"
set "TORCHVISION_VERSION=0.26.0"
REM Pinned to a specific model repo commit (not "main") for reproducibility.
set "CHECKPOINT_REVISION=228d79cb20811466f5c5710aa91f05dabd0b8a14"
set "DEFAULT_CHECKPOINT_URL=https://huggingface.co/Lykon/DreamShaper/resolve/%CHECKPOINT_REVISION%/DreamShaper_8_pruned.safetensors"
set "DEFAULT_CHECKPOINT_NAME=DreamShaper_8_pruned.safetensors"

set "APPDIR=%LOCALAPPDATA%\DaraskAIDiffusion"
set "VENV=%APPDIR%\env"
set "PYTHON_EXE=%VENV%\Scripts\python.exe"
set "COMFY_DIR=%APPDIR%\ComfyUI"
set "COMFY_MAIN=%COMFY_DIR%\main.py"
set "COMFY_PORT=8188"
set "PLUGIN_HOST=127.0.0.1"
set "PLUGIN_PORT=8424"
set "SCRIPT_DIR=%~dp0"
set "COMFY_LOG=%APPDIR%\comfyui.log"
set "SETUP_MARKER=%APPDIR%\.darask_setup"

REM Setup is considered complete only if the marker records exactly this
REM script's expected versions AND a hash of the currently-installed
REM requirements.txt still matches what was recorded when it was written
REM (so a partially-corrupted or hand-edited install is not trusted).
set "REQ_HASH="
if exist "%COMFY_DIR%\requirements.txt" (
    for /f "usebackq delims=" %%h in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -Path '%COMFY_DIR%\requirements.txt').Hash"`) do set "REQ_HASH=%%h"
)
set "EXPECTED_MARKER=comfy=%COMFY_VERSION%;torch=%TORCH_VERSION%;torchvision=%TORCHVISION_VERSION%;reqhash=!REQ_HASH!"
set "CURRENT_MARKER="
if exist "%SETUP_MARKER%" set /p CURRENT_MARKER=<"%SETUP_MARKER%"

if "!CURRENT_MARKER!"=="!EXPECTED_MARKER!" if exist "%COMFY_MAIN%" if exist "%PYTHON_EXE%" goto :run

echo === darask-paint AI Diffusion plugin: first-time setup ===
echo Install location: %APPDIR%
echo This step downloads ComfyUI + PyTorch (roughly 5-8GB total with
echo CUDA, less for CPU-only) and can take a long time depending on
echo your connection.
echo.

where uv >nul 2>nul
if not errorlevel 1 goto :have_uv
echo [1/6] Installing uv package manager...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo ERROR: uv installation failed. Install it manually from https://docs.astral.sh/uv/
    goto :fail
)
:have_uv

echo [2/6] Creating Python environment...
if not exist "%APPDIR%" mkdir "%APPDIR%"
uv venv "%VENV%" --python 3.12
if errorlevel 1 goto :fail

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
echo [3/6] NVIDIA GPU detected - installing PyTorch %TORCH_VERSION% with CUDA 12.8...
echo       NOTE: needs a driver supporting CUDA 12.8+ (Blackwell/Ada/Ampere are fine).
uv pip install --python "%PYTHON_EXE%" torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --torch-backend=cu128
if errorlevel 1 (
    echo ERROR: CUDA PyTorch install failed. Update your NVIDIA driver and retry.
    goto :fail
)
goto :torch_done
:torch_cpu
echo [3/6] No NVIDIA GPU detected - installing CPU PyTorch %TORCH_VERSION%...
echo       NOTE: image generation will be slow on CPU (many minutes per image).
uv pip install --python "%PYTHON_EXE%" torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --torch-backend=cpu
if errorlevel 1 goto :fail
:torch_done

if exist "%COMFY_MAIN%" goto :comfy_installed
echo [4/6] Downloading ComfyUI (%COMFY_VERSION%)...
set "COMFY_ZIP=%APPDIR%\ComfyUI-%COMFY_VERSION%.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%COMFY_REPO_URL%/archive/%COMFY_VERSION%.zip' -OutFile '%COMFY_ZIP%'"
if errorlevel 1 (
    echo ERROR: Failed to download ComfyUI.
    goto :fail
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%COMFY_ZIP%' -DestinationPath '%APPDIR%' -Force"
if errorlevel 1 goto :fail
if exist "%COMFY_DIR%" rmdir /s /q "%COMFY_DIR%"
move "%APPDIR%\ComfyUI-%COMFY_VERSION%" "%COMFY_DIR%" >nul
if errorlevel 1 goto :fail
del "%COMFY_ZIP%" >nul 2>nul

echo       Installing ComfyUI's Python dependencies...
uv pip install --python "%PYTHON_EXE%" -r "%COMFY_DIR%\requirements.txt"
if errorlevel 1 goto :fail
:comfy_installed

if exist "%COMFY_DIR%\models\checkpoints\*.safetensors" goto :model_installed
if exist "%COMFY_DIR%\models\checkpoints\*.ckpt" goto :model_installed
echo.
echo [5/6] No checkpoint model found under:
echo       %COMFY_DIR%\models\checkpoints
echo       You can download a default one now: DreamShaper (SD1.5, ~2GB,
echo       from huggingface.co/Lykon/DreamShaper). This is only needed once.
echo       If you skip this, install any SD1.5/SDXL checkpoint into that
echo       folder yourself before using AI generate/replace.
set /p "DL_MODEL=Download the default model now? [Y/n] "
if /i "!DL_MODEL!"=="n" goto :model_skip
echo       Downloading %DEFAULT_CHECKPOINT_NAME% (this can take a while)...
if not exist "%COMFY_DIR%\models\checkpoints" mkdir "%COMFY_DIR%\models\checkpoints"
set "MODEL_PART=%COMFY_DIR%\models\checkpoints\%DEFAULT_CHECKPOINT_NAME%.part"
if exist "%MODEL_PART%" del "%MODEL_PART%" >nul 2>nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri '%DEFAULT_CHECKPOINT_URL%' -OutFile '%MODEL_PART%'"
if errorlevel 1 (
    echo WARNING: Model download failed. You can add a checkpoint manually later.
    if exist "%MODEL_PART%" del "%MODEL_PART%" >nul 2>nul
) else (
    REM Only becomes the real filename once the download has fully succeeded,
    REM so a checkpoint file is never mistaken for complete while half-written.
    move /y "%MODEL_PART%" "%COMFY_DIR%\models\checkpoints\%DEFAULT_CHECKPOINT_NAME%" >nul
)
goto :model_installed
:model_skip
echo       Skipped. Remember to add a checkpoint before using AI generate/replace.
:model_installed

REM Record setup completion only now that every step above has succeeded.
if exist "%COMFY_DIR%\requirements.txt" (
    for /f "usebackq delims=" %%h in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -Path '%COMFY_DIR%\requirements.txt').Hash"`) do set "REQ_HASH=%%h"
)
> "%SETUP_MARKER%" echo comfy=%COMFY_VERSION%;torch=%TORCH_VERSION%;torchvision=%TORCHVISION_VERSION%;reqhash=!REQ_HASH!

echo.
echo [6/6] Setup finished successfully.
echo.

:run
if not exist "%COMFY_MAIN%" (
    echo ERROR: ComfyUI is not installed ^(missing %COMFY_MAIN%^). Delete
    echo        %APPDIR% and run this script again to reinstall.
    goto :fail
)
REM ComfyUI defaults to CUDA and aborts at startup with "Torch not compiled with
REM CUDA enabled" on a CPU-only PyTorch, so ask the installed torch which it is.
set "COMFY_ARGS="
"%PYTHON_EXE%" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if errorlevel 1 (
    set "COMFY_ARGS=--comfy-arg=--cpu"
    echo   Device:     CPU ^(no usable CUDA in the installed PyTorch^) - generation will be slow
) else (
    echo   Device:     CUDA
)
echo Starting darask-paint AI Diffusion plugin.
echo   ComfyUI:    starting on http://127.0.0.1:%COMFY_PORT% (log: %COMFY_LOG%)
echo   API server: http://%PLUGIN_HOST%:%PLUGIN_PORT%  (darask-paint: "AI seisei / AI chikan (Diffusion)" menus)
echo Close this window (or press Ctrl+C) to stop both ComfyUI and the API server.
echo The first generation after starting can take a while while ComfyUI loads the model.
echo.
REM Extra arguments to this .bat (e.g. "darask-plugin.bat --checkpoint foo.safetensors")
REM are forwarded to darask_server.py as-is.
"%PYTHON_EXE%" "%SCRIPT_DIR%darask_server.py" --port %PLUGIN_PORT% --comfy-port %COMFY_PORT% --comfy-python "%PYTHON_EXE%" --comfy-main "%COMFY_MAIN%" --comfy-log "%COMFY_LOG%" %COMFY_ARGS% %*
goto :eof

:fail
echo.
echo Setup failed. See the messages above for details.
pause
exit /b 1
