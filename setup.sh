#!/bin/bash

# Whisper LoRA Fine-tuning Setup Script
# This script sets up the virtual environment and installs dependencies

echo "=========================================="
echo "WHISPER LORA FINE-TUNING SETUP"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    echo "Please check the requirements.txt file and try again"
    exit 1
fi

echo "✓ Dependencies installed successfully"

# Run setup test
echo "Running setup test..."
python3 test_setup.py

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To start training:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Update DATA_JSON_PATH in train_whisper_lora.py"
echo "3. Run training: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py"
echo ""
echo "For more information, see README.md"
