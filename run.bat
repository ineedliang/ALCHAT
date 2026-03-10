@echo off
title AI Chat
echo Starting AI Chat...
python main.py
if errorlevel 1 (
    echo.
    echo App exited with an error. Did you run setup.bat first?
    pause
)
