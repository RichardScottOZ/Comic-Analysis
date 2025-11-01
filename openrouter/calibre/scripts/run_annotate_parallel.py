@echo off
setlocal enabledelayedexpansion

set SCRIPT_PATH="C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\annotate_page_types.py"
set INPUT_CSV="C:\Users\Richard\OneDrive\GIT\CoMix\key_mapping_report_claude_amazon.csv"
set NUM_SPLITS=8
set BASE_OUTPUT_DIR="C:\Users\Richard\OneDrive\GIT\CoMix"

echo Starting parallel annotation for %NUM_SPLITS% splits...
echo.

rem Calculate the upper bound for the loop (NUM_SPLITS - 1)
set /A UPPER_BOUND=%NUM_SPLITS%-1

for /L %%i in (0,1,%UPPER_BOUND%) do (
    set "OUTPUT_CSV_SPLIT_VAR=%BASE_OUTPUT_DIR%\key_mapping_report_claude_amazon_split%%i_with_page_types.csv"
    echo Launching split %%i...
    start "" cmd /k python %SCRIPT_PATH% --input_csv %INPUT_CSV% --read_vlm_json --num_splits %NUM_SPLITS% --index %%i --output_csv "!OUTPUT_CSV_SPLIT_VAR!"
)

echo.
echo All %NUM_SPLITS% splits have been launched in separate command windows.
echo Please check those windows for individual progress and completion messages.
echo This main window will remain open.
pause
