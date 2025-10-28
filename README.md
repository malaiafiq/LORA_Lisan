# Whisper LoRA Fine-tuning

This project provides a complete setup for fine-tuning OpenAI's Whisper model using LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning on custom speech recognition datasets.

## Features

- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **8-bit Quantization**: Memory-efficient training with optional 8-bit quantization
- **Automatic Data Splitting**: 90:10 train/test split from single JSON file
- **WER Evaluation**: Built-in Word Error Rate calculation during and after training
- **Early Stopping**: Prevents overfitting with automatic training termination (via callback)
- **Progress Table Output**: Clear per-epoch table of Train/Eval losses and WER
- **MPS Support**: Optimized for Apple Silicon (M1/M2) GPUs
- **Flexible Data Format**: Supports both 'text' and 'transcript' keys in JSON

## Setup

### 1. Create Virtual Environment

#### For macOS/Linux:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### For Windows:
```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

### 2. Install Dependencies

#### For macOS/Linux:
```bash
# Install all required packages
pip install -r requirements.txt
```

#### For Windows:
```cmd
# Install all required packages
pip install -r requirements.txt
```

### 3. Prepare Your Data

Create a JSON file with your audio-text pairs in the following format:

```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "text": "Your transcription text here"
  },
  {
    "audio_path": "/path/to/audio2.wav", 
    "transcript": "Alternative key name for transcription"
  }
]
```

**Note**: The script supports both `"text"` and `"transcript"` keys for flexibility.

## Usage

### 1. Update Data Path

Edit the `DATA_JSON_PATH` variable in `train_whisper_lora.py`:

```python
DATA_JSON_PATH = "/path/to/your/dataset.json"  # macOS/Linux
# or
DATA_JSON_PATH = "C:\\path\\to\\your\\dataset.json"  # Windows
```

### 2. Run Training

#### For macOS/Linux:
```bash
# Activate virtual environment
source .venv/bin/activate

# Run training with memory optimization for Apple Silicon
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
```

#### For Windows:
```cmd
# Activate virtual environment
.venv\Scripts\activate

# Run training (Windows doesn't need MPS optimization)
python train_whisper_lora.py
```

## Training Configuration

The script uses the following optimized settings:

- **Model**: `openai/whisper-base` (Malay language)
- **LoRA Rank**: 16 (balanced efficiency and performance)
- **Learning Rate**: 5e-5 (stable training)
- **Batch Size**: 1 (memory efficient)
- **Gradient Accumulation**: 8 steps
- **Epochs**: 10 (maximum, with early stopping)
- **Early Stopping**: 3 evaluations without improvement
- **Evaluation**: Every 25 steps
- **Checkpointing**: Every 25 steps

## Output

### Training Progress

The script now prints a comprehensive progress table after training:

```
🏋️  TRAINING PROGRESS TABLE:
--------------------------------------------------------------------------------
Epoch    Training Loss   Validation Loss  Validation WER
--------------------------------------------------------------------------------
1.0      2.4446          2.1129           2.9520
2.0      2.0675          1.9913           2.9520
3.0      1.8157          1.9153           2.9520
4.0      1.6420          1.8498           3.6531
--------------------------------------------------------------------------------
FINAL    1.6420          1.8498           3.6531
```

During training you will also see step-wise logs; evaluation runs every 25 steps.

### Model Checkpoints

- **Location**: `checkpoints/` directory
- **Best Model**: Automatically loaded at the end
- **Checkpoint Limit**: 3 most recent checkpoints saved

### Final WER Report

After training completes, you'll see:

```
==================================================
COMPUTING FINAL WER METRICS
==================================================

FINAL RESULTS:
Word Error Rate (WER): 0.1234 (12.34%)
Model saved to: checkpoints
==================================================
```

## Early Stopping

Early stopping is configured via the `EarlyStoppingCallback` (not via `Seq2SeqTrainingArguments`). Current defaults:

- Patience: 3 evaluations
- Threshold: 0.001 minimum improvement
- Metric: `eval_loss` (lower is better)

Implementation snippet:

```python
from transformers import EarlyStoppingCallback

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    # ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
)
```

## Memory Optimization

### For Apple Silicon (M1/M2) Macs:
- **MPS Backend**: Automatic GPU acceleration
- **Memory Management**: Optimized batch sizes and data loading
- **8-bit Quantization**: Optional memory reduction (requires CUDA-compatible bitsandbytes)

### For Windows:
- **CUDA Support**: Automatic GPU acceleration if CUDA is available
- **CPU Fallback**: Works on CPU if no GPU is available
- **8-bit Quantization**: Available for CUDA systems with proper bitsandbytes installation

## Troubleshooting

### Common Issues

#### For macOS/Linux:
1. **MPS Out of Memory**:
   ```bash
   # Use memory optimization
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
   ```

2. **Missing Dependencies**:
   ```bash
   # Reinstall requirements
   pip install -r requirements.txt
   ```

#### For Windows:
1. **CUDA Out of Memory**:
   ```cmd
   # Reduce batch size in training script
   # Or use CPU-only mode
   python train_whisper_lora.py
   ```

2. **Python Path Issues**:
   ```cmd
   # Use full path to python
   C:\Python39\python.exe train_whisper_lora.py
   ```

3. **Missing Dependencies**:
   ```cmd
   # Reinstall requirements
   pip install -r requirements.txt
   ```

#### General Issues:
4. **Audio Loading Errors**:
   - Ensure audio files are in supported formats (WAV, MP3, etc.)
   - Check file paths in your JSON are correct
   - Use forward slashes `/` in paths for cross-platform compatibility

### Performance Tips

- **Smaller Dataset**: Start with 50-100 samples for testing
- **Reduce Epochs**: Use 3-5 epochs for initial experiments
- **Monitor Memory**: Watch system memory usage during training
- **Platform-Specific**: Use appropriate commands for your operating system

## File Structure

```
Lora_whisper/
├── train_whisper_lora.py    # Main training script
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── example_data.json       # Sample data format
└── checkpoints/            # Model checkpoints (created during training)
```

## Requirements

### System Requirements:
- **Python**: 3.8+ (Python 3.9+ recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 2GB+ free space for model and dependencies

### Platform-Specific Requirements:

#### For macOS:
- **Apple Silicon**: M1/M2 Mac (recommended for MPS acceleration)
- **Intel Mac**: Supported but may be slower
- **Audio Support**: Built-in audio libraries

#### For Windows:
- **GPU**: CUDA-compatible GPU (NVIDIA) for acceleration
- **CPU**: Modern multi-core processor (Intel/AMD)
- **Audio Support**: May require additional audio codecs

#### For Linux:
- **GPU**: CUDA-compatible GPU (NVIDIA) for acceleration
- **CPU**: Modern multi-core processor
- **Audio Support**: ALSA/PulseAudio for audio processing

### Audio File Requirements:
- **Formats**: WAV, MP3, FLAC, M4A, WMA
- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono or stereo (automatically converted)

## Advanced Configuration

### Custom LoRA Parameters

Edit the LoRA configuration in `train_whisper_lora.py`:

```python
config = LoraConfig(
    r=16,  # Rank (higher = more parameters, lower = more efficient)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,  # Dropout rate
    bias="none",  # Bias adaptation
)
```

### Training Parameters

Modify `Seq2SeqTrainingArguments` for different training strategies:

```python
training_args = Seq2SeqTrainingArguments(
    learning_rate=5e-5,  # Adjust learning rate
    num_train_epochs=10,  # Maximum number of epochs
    per_device_train_batch_size=1,  # Batch size
    gradient_accumulation_steps=8,  # Gradient accumulation
    # Early stopping parameters
    early_stopping_patience=3,  # Stop if no improvement for 3 evaluations
    early_stopping_threshold=0.001,  # Minimum improvement threshold
    # ... other parameters
)
```

### Early Stopping Configuration

The early stopping mechanism helps prevent overfitting:

- **Patience**: Number of evaluations to wait for improvement (default: 3)
- **Threshold**: Minimum improvement required (default: 0.001)
- **Metric**: Uses `eval_loss` to determine improvement
- **Behavior**: Training stops automatically when no improvement is detected

## License

This project is for educational and research purposes. Please respect OpenAI's Whisper model license terms.