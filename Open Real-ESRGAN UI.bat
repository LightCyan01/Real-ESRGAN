@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is required to set up Real-ESRGAN Upscaler.
    echo Install it with:
    echo powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 environment...
    uv venv --python 3.11 || goto fail
)

call :check_deps
if errorlevel 1 (
    call :install_deps || goto fail
    call :check_deps || goto fail
)

".venv\Scripts\python.exe" "ui\app.py"
if errorlevel 1 goto fail
exit /b 0

:check_deps
".venv\Scripts\python.exe" -c "import realesrgan._compat; import torch, torchvision, cv2, PIL, tqdm, basicsr, facexlib, gfpgan, webview, spandrel, realesrgan" >nul 2>nul
exit /b %errorlevel%

:install_deps
echo Installing dependencies. This can take a while the first time...
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || exit /b 1
uv pip install "numpy<2" opencv-python Pillow tqdm cython pywebview spandrel || exit /b 1
uv pip install "basicsr>=1.4.2" facexlib gfpgan --no-build-isolation || exit /b 1
uv pip install -e . || exit /b 1
exit /b 0

:fail
echo.
echo Real-ESRGAN Upscaler could not start. See the error above.
pause
exit /b 1
