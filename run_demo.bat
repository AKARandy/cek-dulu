@echo off
REM cek-dulu demo launcher: starts the Gradio UI and opens your browser.
cd /d "%~dp0"
echo Starting cek-dulu demo (first load takes ~30-60s while models wake up)...
start "" http://127.0.0.1:7860
.venv\Scripts\python demo\app.py
pause
