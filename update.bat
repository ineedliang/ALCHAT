@echo off
title AI Chat - Update Dependencies
echo.
echo Updating all AI Chat dependencies...
echo.
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --quiet
python -m pip install --upgrade -r requirements.txt --quiet
echo.
echo Done. All packages updated.
pause
