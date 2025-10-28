# Quick Start Guide

## 🚀 Get Started in 3 Steps

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
# Run training with memory optimization
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
```

#### For Windows:
```cmd
# Run training
python train_whisper_lora.py
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
Edit `train_whisper_lora.py` line 149:
```python
DATA_JSON_PATH = "/path/to/your/dataset.json"
```

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

### Memory Issues (Apple Silicon)
```bash
# Use memory optimization
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 train_whisper_lora.py
```

### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt
```

### Data Format Issues
- Ensure audio files exist and are accessible
- Check JSON format matches example
- Verify 'audio_path' and 'text'/'transcript' keys

## 📈 Training Parameters

- **Model**: Whisper-base (Malay)
- **Method**: LoRA fine-tuning
- **Epochs**: 10 (maximum, with early stopping)
- **Early Stopping**: 3 evaluations without improvement
- **Learning Rate**: 5e-5
- **Batch Size**: 1 (with gradient accumulation)
- **Evaluation**: Every 25 steps
- **Checkpoints**: Every 25 steps

## 🎯 Expected Results

- **Training Time**: 5-15 minutes (depending on dataset size)
- **Memory Usage**: 4-8GB RAM
- **Output**: Fine-tuned model in `checkpoints/` directory
- **Metrics**: WER (Word Error Rate) reported after training

## 📚 More Information

- **Full Documentation**: See `README.md`
- **Example Data**: See `example_data.json`
- **Setup Issues**: Run `python3 test_setup.py`
