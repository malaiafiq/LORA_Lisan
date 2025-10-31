@echo off
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo.
echo To run the enhanced training script:
echo   python train_whisper_lora_enhanced_simple.py
echo.
echo To test the installation:
echo   python test_simple.py
echo.
cmd /k
