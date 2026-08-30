@echo off
cd /d C:\Users\Richard\OneDrive\GIT\Comic-Analysis
echo Starting Comic VLM Search v2 (Stage 3 + Stage 4 narrative search)...
echo NOTE: Loads Stage 3 model + both Zarr files into RAM on first start.
echo Open your browser at http://127.0.0.1:5003
python -u src\version2\interface\search_server_vlm_v2.py
if errorlevel 1 pause
