# Quick Start Guide

## 🚀 Get Started in 3 Steps

### ⚠️ Important: Platform Requirements

**BitsAndBytesConfig (8-bit quantization) requires Ubuntu/Linux with CUDA support.**

- **Ubuntu/Linux**: Required for 8-bit quantization (bitsandbytes library)
- **Windows**: Script works but will use non-quantized model (more memory)
- **WSL2**: Use Ubuntu distribution for full 8-bit quantization support
- **Automatic Fallback**: Script automatically falls back if 8-bit quantization fails

### 1. Setup Environment

#### For macOS/Linux:
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### For Windows:
```cmd
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Setup

#### For macOS/Linux:
```bash
# Verify everything is working
python3 test_setup.py
```

#### For Windows:
```cmd
# Verify everything is working
python test_setup.py
```

### 3. Start Training

#### For macOS/Linux:
```bash
# Run training with memory optimization (if using Apple Silicon)
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py

# Or with custom dataset path
python3 train_whisper_lora.py --data_json "/path/to/your/dataset.json"
```

#### For Windows:
```cmd
# Run training
python train_whisper_lora.py

# Or with custom dataset path
python train_whisper_lora.py --data_json "D:\path\to\your\dataset.json"
```

#### For WSL:
```bash
# Run training (paths automatically converted)
python3 train_whisper_lora.py

# Or with custom dataset path
python3 train_whisper_lora.py --data_json "/mnt/d/path/to/dataset.json"
```

## 📁 File Structure
```
Lora_whisper/
├── train_whisper_lora.py    # Main training script
├── test_setup.py           # Setup verification
├── requirements.txt           # Dependencies
├── README.md                 # Full documentation
├── example_data.json        # Sample data format
└── checkpoints/             # Model checkpoints (created during training)
```

## ⚙️ Configuration

### Update Data Path

**Option 1: Use Command Line Argument (Recommended)**
```bash
python train_whisper_lora.py --data_json "/path/to/your/dataset.json"
```

**Option 2: Use Default Paths**
The script automatically uses OS-specific default paths:
- **Windows**: `D:\Afiq.hamidon\Lisan V2\organized_dataset.json`
- **Linux/WSL**: `/mnt/d/Afiq.hamidon/Lisan V2/organized_dataset.json`

**Option 3: Edit Script**
Edit `train_whisper_lora.py` in the `main()` function to change default paths.

### Data Format
Your JSON file should look like:
```json
[
  {
    "audio_path": "/path/to/audio1.wav",
    "text": "Your transcription here"
  }
]
```

## 📊 What You'll See

### During Training:
- Training loss and learning rate
- Evaluation metrics every 25 steps
- Progress bars and timing

### After Training:
```
==================================================
COMPUTING FINAL WER METRICS
==================================================

FINAL RESULTS:
Word Error Rate (WER): 0.1234 (12.34%)
Model saved to: checkpoints
==================================================
```

## 🔧 Troubleshooting

### 8-bit Quantization Issues (BitsAndBytesConfig)
**Problem**: "8-bit quantization failed" or ImportError

**Solutions**:
- **Ubuntu/Linux Required**: `bitsandbytes` only works on Linux/Ubuntu
- **Windows Users**: Script will automatically use non-quantized model (more memory usage)
- **WSL2 Users**: Install CUDA in WSL and use Ubuntu distribution
- **Automatic Fallback**: The script handles this gracefully - training continues without quantization

**Example error handling**:
```python
# The script automatically handles this:
try:
    # Try 8-bit quantization
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base", 
        quantization_config=quantization_config,
        device_map="auto"
    )
except ImportError:
    # Falls back to non-quantized model
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base",
        device_map="auto"
    )
```

### Memory Issues (Apple Silicon)
```bash
# Use memory optimization
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt

# Note: bitsandbytes requires Ubuntu/Linux with CUDA
```

### Data Format Issues
- Ensure audio files exist and are accessible
- Check JSON format matches example
- Verify 'audio_path' and 'text'/'transcript' keys

## 📈 Training Parameters

- **Model**: Whisper-base (Malay)
- **Method**: LoRA fine-tuning with 8-bit quantization
- **Epochs**: 100 (maximum, with early stopping)
- **Early Stopping**: 3 epochs without improvement
- **Learning Rate**: 1e-5 (optimized for 8-bit training)
- **Batch Size**: 1 (with gradient accumulation of 8)
- **Evaluation**: Every epoch
- **Checkpoints**: Every epoch (keeps best 3 based on WER)
- **Best Model Metric**: Word Error Rate (WER)
- **Gradient Clipping**: Enabled (max_norm=1.0)
- **Gradient Checkpointing**: Enabled (use_reentrant=False)

## 🎯 Expected Results

- **Training Time**: 5-15 minutes (depending on dataset size)
- **Memory Usage**: 4-8GB RAM
- **Output**: Fine-tuned model in `checkpoints/` directory
- **Metrics**: WER (Word Error Rate) reported after training

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Example Data**: See `example_data.json`
- **Setup Issues**: Run `python3 test_setup.py`
