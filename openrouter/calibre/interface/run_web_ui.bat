@echo off
rem This script starts the visual search interface.

set FLASK_APP=search_server.py

echo Starting the CoMix Visual Search server...
echo Open your web browser and go to http://127.0.0.1:5001

flask run --port=5001
