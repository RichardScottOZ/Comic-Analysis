@echo off
setlocal enabledelayedexpansion

set SCRIPT_PATH="C:\Users\Richard\OneDrive\GIT\CoMix\benchmarks\detections\openrouter\annotate_page_types.py"
set INPUT_CSV="C:\Users\Richard\OneDrive\GIT\CoMix\key_mapping_report_claude_amazon.csv"
set NUM_SPLITS=8
set BASE_OUTPUT_DIR=C:\Users\Richard\OneDrive\GIT\CoMix
set SPLIT_OUTPUT_PREFIX=%BASE_OUTPUT_DIR%\key_mapping_report_claude_amazon_split
set COMBINED_OUTPUT_CSV=%BASE_OUTPUT_DIR%\key_mapping_report_claude_amazon_combined.csv

echo Starting parallel annotation for %NUM_SPLITS% splits...
echo.

rem Calculate the upper bound for the loop (NUM_SPLITS - 1)
set /A UPPER_BOUND=%NUM_SPLITS%-1

for /L %%i in (0,1,%UPPER_BOUND%) do (
    set "OUTPUT_CSV_SPLIT_VAR=%SPLIT_OUTPUT_PREFIX%%%i_with_page_types.csv"
    echo Launching split %%i...
    start "" cmd /k python %SCRIPT_PATH% --input_csv %INPUT_CSV% --read_vlm_json --return_vlm_json --num_splits %NUM_SPLITS% --index %%i --output_csv "!OUTPUT_CSV_SPLIT_VAR!"
)

echo.
echo All %NUM_SPLITS% splits have been launched in separate command windows.
echo ^>^>^> IMPORTANT: Please wait for ALL of these windows to close or indicate completion. ^<^<^< 
echo. 
echo Once all splits are finished, you can run the combination command manually by executing this command:
echo. 
echo   python %SCRIPT_PATH% --input_csv %INPUT_CSV% --num_splits %NUM_SPLITS% --only_combine --split_output_prefix "%SPLIT_OUTPUT_PREFIX%" --output_csv "%COMBINED_OUTPUT_CSV%"
echo. 
echo This main window will remain open.
pause
