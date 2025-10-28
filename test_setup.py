#!/usr/bin/env python3
"""
Test script to verify the Whisper LoRA fine-tuning setup.
Run this before training to ensure everything is configured correctly.
"""

import sys
import torch
import json
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import transformers
        print(f"✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers: {e}")
        return False
    
    try:
        import peft
        print(f"✓ peft {peft.__version__}")
    except ImportError as e:
        print(f"✗ peft: {e}")
        return False
    
    try:
        import librosa
        print(f"✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"✗ librosa: {e}")
        return False
    
    try:
        import evaluate
        print(f"✓ evaluate")
    except ImportError as e:
        print(f"✗ evaluate: {e}")
        return False
    
    try:
        import jiwer
        print(f"✓ jiwer")
    except ImportError as e:
        print(f"✗ jiwer: {e}")
        return False
    
    return True

def test_torch():
    """Test PyTorch and device availability."""
    print("\nTesting PyTorch...")
    
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ MPS available: {torch.backends.mps.is_available()}")
    
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU support detected")
    elif torch.cuda.is_available():
        print("✓ CUDA GPU support detected")
    else:
        print("⚠ No GPU support detected - training will be slower")

def test_data_format():
    """Test if the data JSON file exists and has correct format."""
    print("\nTesting data format...")
    
    data_path = "/Volumes/Lisan/organized_dataset.json"
    
    if not os.path.exists(data_path):
        print(f"✗ Data file not found: {data_path}")
        print("  Please update DATA_JSON_PATH in train_whisper_lora.py")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("✗ Data should be a list of dictionaries")
            return False
        
        if len(data) == 0:
            print("✗ Data file is empty")
            return False
        
        # Check first item
        first_item = data[0]
        if 'audio_path' not in first_item:
            print("✗ Missing 'audio_path' key in data")
            return False
        
        if 'text' not in first_item and 'transcript' not in first_item:
            print("✗ Missing 'text' or 'transcript' key in data")
            return False
        
        print(f"✓ Data file loaded successfully ({len(data)} samples)")
        print(f"✓ Sample keys: {list(first_item.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False

def test_model_loading():
    """Test if we can load the Whisper model."""
    print("\nTesting model loading...")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration, EarlyStoppingCallback
        
        print("Loading Whisper processor...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ms", task="transcribe")
        print("✓ Processor loaded")
        
        print("Loading Whisper model...")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base", device_map="auto")
        print("✓ Model loaded")
        
        print("Testing EarlyStoppingCallback...")
        callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
        print("✓ Early stopping callback created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("WHISPER LORA FINE-TUNING SETUP TEST")
    print("="*50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test PyTorch
    test_torch()
    
    # Test data format
    if not test_data_format():
        all_passed = False
    
    # Test model loading
    if not test_model_loading():
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready for training!")
        print("\nTo start training, run:")
        print("  source .venv/bin/activate")
        print("  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before training")
    print("="*50)

if __name__ == "__main__":
    main()
