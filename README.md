# Whisper LoRA Fine-tuning

This project provides a complete setup for fine-tuning OpenAI's Whisper model using LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning on custom speech recognition datasets.

## Features

- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **8-bit Quantization**: Memory-efficient training with optional 8-bit quantization
- **Automatic Data Splitting**: 90:10 train/test split from single JSON file
- **WER Evaluation**: Built-in Word Error Rate calculation during and after training
- **Early Stopping**: Prevents overfitting with automatic training termination
- **MPS Support**: Optimized for Apple Silicon (M1/M2) GPUs
- **Flexible Data Format**: Supports both 'text' and 'transcript' keys in JSON

## Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
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
DATA_JSON_PATH = "/path/to/your/dataset.json"
```

### 2. Run Training

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training with memory optimization for Apple Silicon
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
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

The script will display:
- Training loss and learning rate
- Evaluation metrics every 25 steps
- Final WER (Word Error Rate) after training completion

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

## Memory Optimization

For Apple Silicon (M1/M2) Macs, the script includes:

- **MPS Backend**: Automatic GPU acceleration
- **Memory Management**: Optimized batch sizes and data loading
- **8-bit Quantization**: Optional memory reduction (requires CUDA-compatible bitsandbytes)

## Troubleshooting

### Common Issues

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

3. **Audio Loading Errors**:
   - Ensure audio files are in supported formats (WAV, MP3, etc.)
   - Check file paths in your JSON are correct

### Performance Tips

- **Smaller Dataset**: Start with 50-100 samples for testing
- **Reduce Epochs**: Use 3-5 epochs for initial experiments
- **Monitor Memory**: Watch system memory usage during training

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

- Python 3.8+
- 8GB+ RAM (16GB+ recommended)
- Apple Silicon Mac (M1/M2) or CUDA-compatible GPU
- Audio files in supported formats (WAV, MP3, FLAC, etc.)

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