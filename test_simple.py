#!/usr/bin/env python3
"""
Simple test script for the enhanced Whisper LoRA training script.
"""

import os
import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"OK PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"X PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"OK Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"X Transformers import failed: {e}")
        return False
    
    try:
        import peft
        print(f"OK PEFT {peft.__version__}")
    except ImportError as e:
        print(f"X PEFT import failed: {e}")
        return False
    
    try:
        import librosa
        print(f"OK Librosa {librosa.__version__}")
    except ImportError as e:
        print(f"X Librosa import failed: {e}")
        return False
    
    try:
        import evaluate
        print(f"OK Evaluate {evaluate.__version__}")
    except ImportError as e:
        print(f"X Evaluate import failed: {e}")
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
        
        print("OK ExecutionTimer class works correctly")
        return True
    except Exception as e:
        print(f"X ExecutionTimer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Enhanced Whisper LoRA Training Script")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("ExecutionTimer Test", test_timer_class),
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
    else:
        print("Some tests failed. Please check the error messages above.")
    
    print("="*60)

if __name__ == "__main__":
    main()
