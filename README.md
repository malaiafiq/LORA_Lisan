# Whisper LoRA Fine-tuning Script

This repository contains a production-ready Whisper LoRA fine-tuning script with comprehensive features for training Whisper models on custom datasets using Low-Rank Adaptation (LoRA).

## Features

### 🚀 Enhanced Training Features
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **8-bit Quantization**: Memory-efficient training with BitsAndBytesConfig (⚠️ requires Ubuntu/Linux with CUDA)
- **Mixed Precision**: FP16 training for faster training and reduced memory usage
- **Early Stopping**: Prevents overfitting with configurable patience
- **Gradient Checkpointing**: Memory-efficient training with `use_reentrant=False`
- **Gradient Clipping**: Prevents NaN gradients with `max_grad_norm=1.0`
- **Cross-OS Support**: Works on Windows and Linux/WSL with automatic path normalization
- **Automatic Fallback**: Falls back to non-quantized model if 8-bit quantization unavailable

### 📊 Comprehensive Monitoring
- **Progress Table**: Real-time training progress visualization with epoch-by-epoch metrics
- **Performance Metrics**: WER (Word Error Rate) tracking and comparison
- **Best Model Selection**: Automatically selects best model based on WER
- **Data Validation**: Automatic checks for NaN/Inf values and empty data

### 🎯 Training Progress Tracking
- **Epoch-by-Epoch Metrics**: Training loss, validation loss, and WER tracking
- **Validation WER**: Computed during evaluation for each epoch
- **Training Statistics**: Comprehensive training summary
- **Real-time Updates**: Progress table displayed after each epoch

## Installation

### ⚠️ Important: Platform Requirements

**BitsAndBytesConfig (8-bit quantization) requires Ubuntu/Linux with CUDA support.**

The 8-bit quantization feature uses `bitsandbytes` library which:
- **Requires Linux/Ubuntu** (does not work on Windows natively)
- **Requires CUDA** for GPU acceleration
- Falls back to non-quantized model if 8-bit quantization fails

**For Windows users:**
- The script will automatically fall back to non-quantized model if BitsAndBytesConfig fails
- Training will work but will use more memory
- Consider using WSL2 (Windows Subsystem for Linux) with Ubuntu for full 8-bit quantization support

**For WSL users:**
- Ensure you have CUDA installed in WSL
- Use Ubuntu distribution in WSL for best compatibility

### 1. Set up Virtual Environment

#### For Ubuntu/Linux:
```bash
# Navigate to the project directory
cd /path/to/LORA_Lisan-main

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### For Windows (without 8-bit quantization):
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

#### For WSL (Ubuntu in Windows):
```bash
# Navigate to the project directory
cd /mnt/d/Afiq.hamidon/Lisan\ V2/Lora-Lisan/LORA_Lisan-main

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch torchvision torchaudio transformers datasets peft accelerate bitsandbytes evaluate librosa scikit-learn tqdm psutil

# Note: bitsandbytes requires CUDA and works best on Ubuntu/Linux
```

## Usage

### Basic Usage

#### Windows:
```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the training script
python train_whisper_lora.py
```

#### Linux/WSL:
```bash
# Activate virtual environment
source venv/bin/activate

# Run the training script
python3 train_whisper_lora.py
```

### Command Line Arguments
The script supports command-line arguments for dataset path:

```bash
# Use custom dataset path
python train_whisper_lora.py --data_json "/path/to/your/dataset.json"

# Use default path (OS-specific)
python train_whisper_lora.py
```

### Configuration
The script automatically detects the operating system and uses appropriate default paths:
- **Windows**: `D:\Afiq.hamidon\Lisan V2\organized_dataset.json`
- **Linux/WSL**: `/mnt/d/Afiq.hamidon/Lisan V2/organized_dataset.json`

You can override this using the `--data_json` argument.

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

### Main Script
- `train_whisper_lora.py` - Main training script with all features:
  - LoRA fine-tuning with 8-bit quantization
  - Gradient checkpointing and clipping
  - Cross-OS path support
  - WER-based best model selection
  - Real-time progress tracking
  - Data validation

### Configuration Files
- `requirements.txt` - Python dependencies
- `README.md` - This documentation
- `QUICK_START.md` - Quick start guide

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
- **Learning Rate**: 1e-5 (reduced for stability with 8-bit quantization)
- **Batch Size**: 1 (with gradient accumulation of 8)
- **Epochs**: 100 (with early stopping)
- **Evaluation Strategy**: `epoch` (evaluates after each epoch)
- **Save Strategy**: `epoch` (saves after each epoch)
- **Best Model Metric**: `eval_wer` (Word Error Rate)
- **Gradient Clipping**: `max_grad_norm=1.0`
- **Gradient Checkpointing**: Enabled with `use_reentrant=False`

### Key Features
- **Attention Masks**: Explicitly provided for both encoder and decoder
- **Data Validation**: Checks for NaN/Inf values and empty data
- **Cross-OS Path Support**: Automatically handles Windows and Linux/WSL paths
- **WER Computation**: Computed during evaluation for best model selection

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
    output_dir="checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,  # Reduced for 8-bit stability
    num_train_epochs=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_wer",
    max_grad_norm=1.0,  # Gradient clipping
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
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
The script displays a progress table after each epoch:
```
TRAINING PROGRESS TABLE:
--------------------------------------------------------------------------------
Epoch    Training Loss   Validation Loss  Validation WER  
--------------------------------------------------------------------------------
1        2.3456         2.1234          0.4567          
2        1.9876         1.8765          0.3456          
3        1.6543         1.5432          0.2345          
...
--------------------------------------------------------------------------------
FINAL    0.1234         0.2345          0.1234          
--------------------------------------------------------------------------------
```

### Performance Metrics
After training completes, the script displays final WER metrics computed on the **test dataset (held-out 10%)** with **correct transcriptions (ground truth)** from the dataset:

```
COMPUTING FINAL WER METRICS
==================================================
Computing WER metrics...
Evaluating: 100%|████████████████████| 6/6 [00:21<00:00, 3.61s/it]

FINAL RESULTS:
Word Error Rate (WER): 0.1234 (12.34%)
Model saved to: checkpoints
==================================================
```

**Note on WER Computation:**
- The final WER is computed on the **test dataset** (10% held-out from training)
- Uses **ground truth transcriptions** from the dataset for comparison
- Uses the same generation parameters as validation during training (`max_length=128`)
- The final WER should align with the validation WER shown in the progress table for the last epoch

## Troubleshooting

### Common Issues

1. **BitsAndBytesConfig / 8-bit Quantization Issues**:
   - **Requires Ubuntu/Linux**: `bitsandbytes` library does not work natively on Windows
   - **Requires CUDA**: GPU with CUDA support is needed for 8-bit quantization
   - **Automatic Fallback**: The script automatically falls back to non-quantized model if 8-bit fails
   - **Windows Users**: Use WSL2 with Ubuntu for full 8-bit quantization support
   - **Error Message**: If you see "8-bit quantization failed", the script will continue without quantization

2. **NaN Gradients**: 
   - Gradient clipping is enabled by default (`max_grad_norm=1.0`)
   - Learning rate is reduced to `1e-5` for stability
   - Data validation checks for NaN/Inf values

3. **Memory Issues**: 
   - 8-bit quantization is enabled by default (requires Ubuntu/Linux with CUDA)
   - Falls back to full precision if quantization unavailable
   - Gradient checkpointing is enabled
   - Batch size is set to 1 with gradient accumulation

4. **Path Issues (WSL/Linux)**:
   - The script automatically converts Windows paths to WSL paths
   - Use `--data_json` to specify custom paths

5. **CUDA Issues**: 
   - The script automatically falls back to CPU if CUDA is not available
   - Uses `device_map="auto"` for automatic device placement
   - 8-bit quantization requires CUDA (works on Ubuntu/Linux only)

6. **Dataset Loading**: 
   - Ensure your JSON file has the correct format
   - Check that audio files exist and are accessible
   - Data validation will catch empty or invalid entries

7. **WER Metric Errors**:
   - The script handles variable-length sequences automatically
   - Empty predictions are handled gracefully

### File Format Issues
- Ensure audio files exist at specified paths
- Check JSON format matches expected structure
- Verify 'audio_path' and 'text'/'transcript' keys are present

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster training
2. **Memory Management**: Use 8-bit quantization for memory efficiency
3. **Batch Size**: Adjust based on available memory
4. **Gradient Accumulation**: Use to simulate larger batch sizes
5. **Early Stopping**: Prevents overfitting and saves time

## File Structure

```
LORA_Lisan-main/
├── train_whisper_lora.py     # Main training script
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── QUICK_START.md            # Quick start guide
├── checkpoints/              # Model checkpoints (created during training)
└── venv/                     # Virtual environment
```

## Technical Details

### Cross-OS Path Support
The script includes automatic path normalization for Windows and Linux/WSL:
- Windows paths: `D:\path\to\file.mp3`
- WSL paths: `/mnt/d/path/to/file.mp3`
- Automatic conversion based on OS detection

### Attention Masks
- **Encoder**: Attention mask created for `input_features` based on actual feature lengths
- **Decoder**: Attention mask provided for labels to handle padding correctly
- Ensures proper handling of variable-length sequences

### Data Validation
Automatic checks are performed:
- NaN/Inf detection in audio data
- Empty audio file detection
- Empty label detection
- Invalid input_features detection

### Gradient Stability
- Gradient clipping: `max_grad_norm=1.0`
- Reduced learning rate: `1e-5` (for 8-bit quantization stability)
- Gradient checkpointing: Enabled with `use_reentrant=False`
- NaN/Inf filtering: Enabled in logging

## Contributing

Feel free to contribute improvements to the training script, add new features, or fix issues.

## License

This project follows the same license as the original Whisper model and Hugging Face Transformers library.