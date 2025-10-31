# Enhanced Whisper LoRA Fine-tuning Script

This repository contains an enhanced version of the Whisper LoRA fine-tuning script with comprehensive timing, progress tracking, and performance metrics based on the reference implementation from `whisper_finetune_with_timing.py`.

## Features

### 🚀 Enhanced Training Features
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **8-bit Quantization**: Memory-efficient training with BitsAndBytesConfig
- **Mixed Precision**: FP16 training for faster training and reduced memory usage
- **Early Stopping**: Prevents overfitting with configurable patience

### 📊 Comprehensive Monitoring
- **Execution Timer**: Detailed timing for each training step
- **Memory Monitoring**: Real-time memory usage tracking
- **Progress Table**: Comprehensive training progress visualization
- **Performance Metrics**: WER (Word Error Rate) tracking and comparison
- **Baseline Evaluation**: Pre-training model performance assessment

### 🎯 Training Progress Tracking
- **Real-time Progress**: Step-by-step execution logging
- **Epoch-by-Epoch Metrics**: Training loss, validation loss, and WER tracking
- **Performance Comparison**: Baseline vs final model performance
- **Training Statistics**: Comprehensive training summary

## Installation

### 1. Set up Virtual Environment
```bash
# Navigate to the project directory
cd "D:\Afiq.hamidon\Lisan V2\Lora-Lisan\LORA_Lisan-main"

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Or for Command Prompt
.\venv\Scripts\activate.bat
```

### 2. Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio transformers datasets peft accelerate bitsandbytes evaluate librosa scikit-learn tqdm psutil
```

## Usage

### Basic Usage
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the enhanced training script
python train_whisper_lora_enhanced_simple.py
```

### Configuration
Before running, update the configuration in the script:

```python
# Update these paths in the main() function
DATA_JSON_PATH = "D:\Afiq.hamidon\Lisan V2\organized_dataset.json"  # Your dataset path
OUTPUT_DIR = "D:/Afiq.hamidon/Lisan V2/Fin_tune_model"  # Output directory
```

### Dataset Format
Your dataset should be a JSON file with the following structure:
```json
[
    {
        "audio_path": "path/to/audio1.wav",
        "text": "Transcription text for audio1"
    },
    {
        "audio_path": "path/to/audio2.wav", 
        "text": "Transcription text for audio2"
    }
]
```

## Scripts Overview

### Main Scripts
- `train_whisper_lora_enhanced_simple.py` - Main enhanced training script (Windows-compatible)
- `train_whisper_lora_enhanced.py` - Original enhanced script with Unicode characters
- `train_whisper_lora.py` - Original LoRA training script

### Test Scripts
- `test_simple.py` - Simple test script for basic functionality
- `test_enhanced_training.py` - Comprehensive test script

### Configuration Files
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Key Features Implementation

### 1. ExecutionTimer Class
```python
timer = ExecutionTimer()
timer.start_step("Loading Data")
# ... your code ...
timer.end_step()
timer.print_summary()
```

### 2. TrainingProgressCallback
```python
progress_callback = TrainingProgressCallback()
progress_callback.set_processor_and_dataset(processor, eval_dataset)
progress_callback.set_baseline_wer(baseline_wer)
```

### 3. Progress Table Display
The script automatically displays:
- Epoch-by-epoch training progress
- Training and validation losses
- WER metrics
- Performance comparison (baseline vs final)

### 4. Performance Metrics
- **WER Calculation**: Word Error Rate for model evaluation
- **Baseline Assessment**: Pre-training model performance
- **Improvement Tracking**: Performance improvement over training
- **Memory Usage**: Real-time memory monitoring

## Training Configuration

### Default Parameters
- **Model**: `openai/whisper-base`
- **Language**: `ms` (Malay)
- **Task**: `transcribe`
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Learning Rate**: 5e-5
- **Batch Size**: 1 (with gradient accumulation)
- **Epochs**: 10
- **Evaluation Steps**: 25

### Customization
You can modify these parameters in the `main()` function:

```python
# LoRA Configuration
config = LoraConfig(
    r=16,  # Rank of LoRA decomposition
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias adaptation
)

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=10,
    # ... other parameters
)
```

## Output

### Training Progress
The script provides real-time feedback:
```
[START] Loading and splitting dataset
Time: 08:51:34
[DONE] Loading and splitting dataset
Duration: 2.3s
Memory: 512.4 MB
```

### Progress Table
```
TRAINING PROGRESS TABLE
================================================================================
Epoch    Training Loss   Validation Loss  Validation WER  
--------------------------------------------------------------------------------
1.0      2.3456         2.1234          0.4567          
2.0      1.9876         1.8765          0.3456          
...
FINAL    0.1234         0.2345          0.1234          
```

### Performance Metrics
```
PERFORMANCE METRICS
================================================================================

PERFORMANCE METRICS:
----------------------------------------
Word Error Rate (WER):
  * Baseline:  0.4567
  * Final:     0.1234
  * Change:    +0.3333 (Improved)

TRAINING STATISTICS:
----------------------------------------
Total Epochs: 10
Total Steps: 150
Evaluation Epochs: 10
```

## Troubleshooting

### Common Issues

1. **Unicode Errors**: Use `train_whisper_lora_enhanced_simple.py` for Windows compatibility
2. **Memory Issues**: Reduce batch size or enable gradient accumulation
3. **CUDA Issues**: The script automatically falls back to CPU if CUDA is not available
4. **Dataset Loading**: Ensure your JSON file has the correct format

### Testing
Run the test script to verify installation:
```bash
python test_simple.py
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Memory Management**: Use 8-bit quantization for memory efficiency
3. **Batch Size**: Adjust based on available memory
4. **Gradient Accumulation**: Use to simulate larger batch sizes
5. **Early Stopping**: Prevents overfitting and saves time

## File Structure

```
LORA_Lisan-main/
├── train_whisper_lora_enhanced_simple.py  # Main enhanced script
├── train_whisper_lora_enhanced.py         # Original enhanced script
├── train_whisper_lora.py                  # Original LoRA script
├── test_simple.py                         # Simple test script
├── test_enhanced_training.py              # Comprehensive test script
├── requirements.txt                       # Dependencies
├── README.md                              # This file
└── venv/                                  # Virtual environment
```

## Contributing

Feel free to contribute improvements to the training script, add new features, or fix issues.

## License

This project follows the same license as the original Whisper model and Hugging Face Transformers library.