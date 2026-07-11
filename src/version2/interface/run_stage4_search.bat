@echo off
cd /d C:\Users\Richard\OneDrive\GIT\Comic-Analysis
echo Starting Comic Stage 4 Semantic Search server...
echo NOTE: First startup may take a while because it loads the model and the full Stage 4 Zarr into RAM.
echo Open your browser at http://127.0.0.1:5004
python -u src\version2\interface\search_server_stage4.py
if errorlevel 1 pause
