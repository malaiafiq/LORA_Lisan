#!/usr/bin/env python3
"""
Test script for the enhanced Whisper LoRA training script.
This script tests the basic functionality without running the full training.
"""

import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Create a small test dataset for testing purposes."""
    test_data = []
    
    # Create a temporary directory for test audio files
    temp_dir = tempfile.mkdtemp()
    
    # Create some dummy audio files and corresponding text
    for i in range(5):
        # Create a dummy audio file (just a numpy array saved as .npy)
        audio_data = np.random.randn(16000)  # 1 second of audio at 16kHz
        audio_path = os.path.join(temp_dir, f"test_audio_{i}.npy")
        np.save(audio_path, audio_data)
        
        # Create corresponding text
        text = f"This is test audio file number {i} for testing purposes."
        
        test_data.append({
            "audio_path": audio_path,
            "text": text
        })
    
    # Save test data to JSON
    test_json_path = os.path.join(temp_dir, "test_dataset.json")
    with open(test_json_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    return test_json_path, temp_dir

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import peft
        print(f"✅ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"❌ PEFT import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        return False
    
    try:
        import evaluate
        print(f"✅ Evaluate {evaluate.__version__}")
    except ImportError as e:
        print(f"❌ Evaluate import failed: {e}")
        return False
    
    return True

def test_timer_class():
    """Test the ExecutionTimer class."""
    print("\nTesting ExecutionTimer class...")
    
    try:
        from train_whisper_lora_enhanced import ExecutionTimer
        
        timer = ExecutionTimer()
        timer.start_step("Test Step")
        import time
        time.sleep(0.1)  # Simulate some work
        timer.end_step()
        
        print("✅ ExecutionTimer class works correctly")
        return True
    except Exception as e:
        print(f"❌ ExecutionTimer test failed: {e}")
        return False

def test_progress_callback():
    """Test the TrainingProgressCallback class."""
    print("\nTesting TrainingProgressCallback class...")
    
    try:
        from train_whisper_lora_enhanced import TrainingProgressCallback
        
        callback = TrainingProgressCallback()
        callback.set_baseline_wer(0.5)
        callback.set_processor_and_dataset(None, [])
        
        print("✅ TrainingProgressCallback class works correctly")
        return True
    except Exception as e:
        print(f"❌ TrainingProgressCallback test failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    try:
        from train_whisper_lora_enhanced import load_and_split_data
        
        # Create test data
        test_json_path, temp_dir = create_test_data()
        
        # Test loading
        train_data, test_data = load_and_split_data(test_json_path, test_size=0.2)
        
        print(f"✅ Data loading works: {len(train_data)} train, {len(test_data)} test")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_loading():
    """Test model loading (without quantization to avoid issues)."""
    print("\nTesting model loading...")
    
    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        # Test processor loading
        processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ms", task="transcribe")
        print("✅ Processor loaded successfully")
        
        # Test model loading (without quantization for testing)
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        print("✅ Model loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced Whisper LoRA Training Script")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("ExecutionTimer Test", test_timer_class),
        ("ProgressCallback Test", test_progress_callback),
        ("Data Loading Test", test_data_loading),
        ("Model Loading Test", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"X {test_name} failed")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The enhanced training script is ready to use.")
        print("\nTo run the training script:")
        print("1. Activate the virtual environment: .\\venv\\Scripts\\Activate.ps1")
        print("2. Update the DATA_JSON_PATH in the script to point to your dataset")
        print("3. Run: python train_whisper_lora_enhanced.py")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main()
