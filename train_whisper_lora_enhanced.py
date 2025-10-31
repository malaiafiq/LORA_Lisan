#!/usr/bin/env python3
"""
Enhanced Whisper LoRA Fine-tuning Script with Detailed Timing and Progress Tracking
=================================================================================

This script fine-tunes OpenAI's Whisper model using LoRA (Low-Rank Adaptation) with:
- Detailed execution timing for each step
- Comprehensive training progress table
- Performance metrics tracking
- Baseline model evaluation
- Memory usage monitoring
- Step-by-step execution logging

Features:
- LoRA fine-tuning for efficient training
- Detailed timing for each execution step
- Progress tracking with estimated completion times
- Memory usage monitoring
- Step-by-step execution logging
- Comprehensive performance metrics
"""

import os
import json
import warnings
import time
import psutil
import numpy as np
import torch
import librosa
import evaluate
from tqdm import tqdm

# Windows multiprocessing fix
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

from transformers import (
    WhisperForConditionalGeneration, 
    WhisperProcessor, 
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    EarlyStoppingCallback, 
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Any, Union
from sklearn.model_selection import train_test_split

# Suppress warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load Deprecated")
warnings.filterwarnings("ignore", message="Failed to find CUDA")
warnings.filterwarnings("ignore", category=UserWarning, module="triton")

class ExecutionTimer:
    """Class to track execution time and provide detailed timing information."""
    
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None
        self.step_start = None
        
    def start_step(self, step_name):
        """Start timing a new step."""
        if self.current_step:
            self.end_step()
        
        self.current_step = step_name
        self.step_start = time.time()
        print(f"\nStarting: {step_name}")
        print(f"Time: {time.strftime('%H:%M:%S')}")
        
    def end_step(self):
        """End timing the current step."""
        if self.current_step and self.step_start:
            duration = time.time() - self.step_start
            self.step_times[self.current_step] = duration
            
            # Format duration
            if duration < 60:
                duration_str = f"{duration:.1f}s"
            elif duration < 3600:
                duration_str = f"{duration/60:.1f}m"
            else:
                duration_str = f"{duration/3600:.1f}h"
            
            print(f"Completed: {self.current_step}")
            print(f"Duration: {duration_str}")
            print(f"Memory: {self.get_memory_usage()}")
            
        self.current_step = None
        self.step_start = None
        
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    
    def get_total_time(self):
        """Get total execution time."""
        total = time.time() - self.start_time
        if total < 60:
            return f"{total:.1f}s"
        elif total < 3600:
            return f"{total/60:.1f}m"
        else:
            return f"{total/3600:.1f}h"
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*60)
        print("📊 EXECUTION SUMMARY")
        print("="*60)
        
        total_time = time.time() - self.start_time
        
        for step, duration in self.step_times.items():
            percentage = (duration / total_time) * 100
            print(f"{step:<30} {duration:>8.1f}s ({percentage:>5.1f}%)")
        
        print("-" * 60)
        print(f"{'TOTAL TIME':<30} {total_time:>8.1f}s")
        print("="*60)

class TrainingProgressCallback(TrainerCallback):
    """Enhanced callback to track and display training progress in a formatted table."""
    
    def __init__(self):
        self.training_logs = []
        self.eval_logs = []
        self.wer_metric = evaluate.load("wer")
        self.baseline_wer = None
        self.final_wer = None
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Capture training and evaluation logs."""
        if logs is not None:
            if 'train_loss' in logs:
                self.training_logs.append({
                    'epoch': logs.get('epoch', 0),
                    'train_loss': logs.get('train_loss', 0),
                    'step': logs.get('step', 0)
                })
            if 'eval_loss' in logs:
                self.eval_logs.append({
                    'epoch': logs.get('epoch', 0),
                    'eval_loss': logs.get('eval_loss', 0),
                    'eval_wer': logs.get('eval_wer', 0),
                    'step': logs.get('step', 0)
                })
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Compute WER during evaluation."""
        if model is not None and hasattr(self, 'processor') and hasattr(self, 'eval_dataset'):
            try:
                device = next(model.parameters()).device
                wer = self._compute_wer_during_eval(model, device)
                if logs is not None:
                    logs['eval_wer'] = wer
                    self.final_wer = wer
            except Exception as e:
                print(f"Error computing WER during evaluation: {e}")
                if logs is not None:
                    logs['eval_wer'] = 0.0
    
    def _compute_wer_during_eval(self, model, device):
        """Compute WER during evaluation (simplified version)."""
        try:
            model.eval()
            predictions = []
            references = []
            
            # Sample a few examples for WER computation during training
            sample_size = min(5, len(self.eval_dataset))
            sample_indices = np.random.choice(len(self.eval_dataset), sample_size, replace=False)
            
            with torch.no_grad():
                for idx in sample_indices:
                    item = self.eval_dataset[idx]
                    audio_path = item['audio_path']
                    reference_text = item.get('text', item.get('transcript', ''))
                    
                    if not reference_text:
                        continue
                        
                    try:
                        # Load and process audio
                        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                        input_features = self.processor.feature_extractor(
                            audio_array,
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_features.to(device)
                        
                        # Generate prediction
                        predicted_ids = model.generate(input_features, max_length=128)
                        prediction = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        
                        predictions.append(prediction)
                        references.append(reference_text)
                        
                    except Exception as e:
                        continue
            
            # Calculate WER
            if len(predictions) > 0 and len(references) > 0:
                wer = self.wer_metric.compute(predictions=predictions, references=references)
                return wer
            else:
                return 0.0
        except Exception as e:
            return 0.0
        finally:
            model.train()
    
    def set_processor_and_dataset(self, processor, eval_dataset):
        """Set processor and eval dataset for WER computation."""
        self.processor = processor
        self.eval_dataset = eval_dataset
    
    def set_baseline_wer(self, baseline_wer):
        """Set baseline WER for comparison."""
        self.baseline_wer = baseline_wer
    
    def display_progress_table(self):
        """Display the comprehensive training progress table."""
        print("\n" + "="*80)
        print("🏋️  TRAINING PROGRESS TABLE")
        print("="*80)
        print(f"{'Epoch':<8} {'Training Loss':<15} {'Validation Loss':<16} {'Validation WER':<15}")
        print("-" * 80)
        
        # Group logs by epoch
        epoch_data = {}
        for log in self.training_logs:
            epoch = log['epoch']
            if epoch not in epoch_data:
                epoch_data[epoch] = {'train_loss': log['train_loss']}
        
        for log in self.eval_logs:
            epoch = log['epoch']
            if epoch in epoch_data:
                epoch_data[epoch]['eval_loss'] = log['eval_loss']
                epoch_data[epoch]['eval_wer'] = log.get('eval_wer', 0.0)
        
        # Display progress for each epoch
        for epoch in sorted(epoch_data.keys()):
            data = epoch_data[epoch]
            train_loss = data.get('train_loss', 0.0)
            eval_loss = data.get('eval_loss', 0.0)
            eval_wer = data.get('eval_wer', 0.0)
            
            print(f"{epoch:<8.1f} {train_loss:<15.4f} {eval_loss:<16.4f} {eval_wer:<15.4f}")
        
        # Display final results
        if epoch_data:
            final_epoch = max(epoch_data.keys())
            final_data = epoch_data[final_epoch]
            print("-" * 80)
            print(f"{'FINAL':<8} {final_data.get('train_loss', 0.0):<15.4f} {final_data.get('eval_loss', 0.0):<16.4f} {final_data.get('eval_wer', 0.0):<15.4f}")
            print("-" * 80)
    
    def display_performance_metrics(self):
        """Display comprehensive performance metrics."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE METRICS")
        print("="*80)
        
        # Performance Metrics
        print(f"\n📊 PERFORMANCE METRICS:")
        print("-" * 40)
        
        if self.baseline_wer is not None and self.final_wer is not None:
            wer_improvement = self.baseline_wer - self.final_wer
            
            print(f"Word Error Rate (WER):")
            print(f"  • Baseline:  {self.baseline_wer:.4f}")
            print(f"  • Final:     {self.final_wer:.4f}")
            print(f"  • Change:    {wer_improvement:+.4f} ({'✅ Improved' if wer_improvement > 0 else '❌ Degraded'})")
        else:
            print(f"Word Error Rate (WER):")
            if self.final_wer is not None:
                print(f"  • Final:     {self.final_wer:.4f}")
            else:
                print(f"  • Final:     Not available")
        
        # Training Statistics
        print(f"\n📈 TRAINING STATISTICS:")
        print("-" * 40)
        
        if self.training_logs:
            total_epochs = len(set(log['epoch'] for log in self.training_logs))
            total_steps = len(self.training_logs)
            print(f"Total Epochs: {total_epochs}")
            print(f"Total Steps: {total_steps}")
        
        if self.eval_logs:
            eval_epochs = len(set(log['epoch'] for log in self.eval_logs))
            print(f"Evaluation Epochs: {eval_epochs}")
        
        print("="*80)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for sequence-to-sequence speech tasks using Whisper.

    This collator dynamically pads both the input audio features and the target text tokens
    to the maximum length in a batch, making it compatible with variable-length input/output sequences.

    Attributes:
        processor (Any): A Hugging Face `WhisperProcessor` that includes both a feature extractor
                         for audio and a tokenizer for text.

    Methods:
        __call__(features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            Pads and collates a batch of audio-text pairs for model input.
    """

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        Pads input audio features and target text labels for a batch of samples.

        Args:
            features (List[Dict]): Each item in the list is a dictionary with:
                - 'input_features': Audio features (from spectrogram extraction)
                - 'labels': Tokenized text labels

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_features': Padded audio features
                - 'labels': Padded and masked labels (with padding tokens replaced by -100)
        """

        # Pad audio features
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=True,
            return_tensors="pt"
        )

        # Pad text labels
        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=True,
            return_tensors="pt"
        )

        # Replace padding token IDs with -100 so they are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Optionally remove BOS token if present at the beginning
        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

class AudioTextDataset(Dataset):
    """
    Dataset for audio-text pairs for Whisper fine-tuning.
    """
    def __init__(self, data: List[Dict], processor):
        self.processor = processor
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio file
        audio_path = item['audio_path']
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Process audio to get input features
        input_features = self.processor.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]
        
        # Process text to get labels - handle both 'text' and 'transcript' keys
        text = item.get('text', item.get('transcript', ''))
        if not text:
            raise ValueError(f"Neither 'text' nor 'transcript' key found in item: {item.keys()}")
        labels = self.processor.tokenizer(text).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }

def load_and_split_data(json_path: str, test_size: float = 0.1, random_state: int = 42):
    """
    Load data from JSON file and split into train/test sets.
    
    Args:
        json_path (str): Path to the JSON file containing audio-text pairs
        test_size (float): Proportion of data to use for testing (default: 0.1 for 90:10 split)
        random_state (int): Random seed for reproducible splits
    
    Returns:
        tuple: (train_data, test_data) - Lists of dictionaries for training and testing
    """
    # Load the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Split the data into train and test sets
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Loaded {len(data)} total samples")
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Split ratio: {len(train_data)/len(data):.1%} train, {len(test_data)/len(data):.1%} test")
    
    return train_data, test_data

def compute_wer_metrics(model, processor, eval_dataset, device):
    """
    Compute WER (Word Error Rate) metrics for evaluation.
    
    Args:
        model: The trained model
        processor: Whisper processor
        eval_dataset: Evaluation dataset
        device: Device to run inference on
    
    Returns:
        float: WER score
    """
    wer_metric = evaluate.load("wer")
    
    model.eval()
    predictions = []
    references = []
    
    print("Computing WER metrics...")
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(eval_dataset, desc="Evaluating")):
            # Get audio and text
            audio_path = item['audio_path']
            reference_text = item.get('text', item.get('transcript', ''))
            
            if not reference_text:
                continue
                
            try:
                # Load and process audio
                audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                input_features = processor.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                
                # Generate prediction
                predicted_ids = model.generate(input_features)
                prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                predictions.append(prediction)
                references.append(reference_text)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    # Calculate WER
    if len(predictions) > 0 and len(references) > 0:
        wer = wer_metric.compute(predictions=predictions, references=references)
        return wer
    else:
        return 1.0  # Return worst possible WER if no valid predictions

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    This is called by the trainer during evaluation.
    """
    predictions, labels = eval_pred
    
    # Return empty dict for now - WER will be computed by the callback
    # The trainer will use this for logging
    return {}

def evaluate_baseline_model(model, processor, eval_dataset, device):
    """Evaluate the model before training to get baseline WER."""
    print("📊 Evaluating baseline model performance...")
    
    # Load evaluation metrics
    wer_metric = evaluate.load("wer")
    
    predictions = []
    references = []
    
    # Evaluate on validation set (use subset for baseline)
    sample_size = min(10, len(eval_dataset))  # Use smaller sample for baseline
    print(f"📊 Evaluating on {sample_size} validation samples for baseline...")
    
    sample_indices = np.random.choice(len(eval_dataset), sample_size, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            if i % 2 == 0:  # Progress indicator every 2 samples
                print(f"   Processing sample {i+1}/{sample_size}...")
            
            item = eval_dataset[idx]
            audio_path = item['audio_path']
            reference_text = item.get('text', item.get('transcript', ''))
            
            if not reference_text:
                continue
                
            try:
                # Load and process audio
                audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                input_features = processor.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                
                # Generate prediction
                predicted_ids = model.generate(input_features, max_length=128)
                prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                predictions.append(prediction)
                references.append(reference_text)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    # Calculate WER
    if len(predictions) > 0 and len(references) > 0:
        wer = wer_metric.compute(predictions=predictions, references=references)
        return {"wer": wer}
    else:
        return {"wer": 1.0}

def main():
    """Main training function with detailed timing and progress tracking."""
    timer = ExecutionTimer()
    
    print("🚀 Starting Enhanced Whisper LoRA Fine-tuning with Detailed Timing")
    print("="*80)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 Initial memory: {timer.get_memory_usage()}")
    print("="*80)
    
    # Configuration
    DATA_JSON_PATH = "D:\Afiq.hamidon\Lisan V2\organized_dataset.json"
    OUTPUT_DIR = "D:/Afiq.hamidon/Lisan V2/Fin_tune_model"
    
    # Set random seeds for reproducibility
    timer.start_step("Setting random seeds")
    torch.manual_seed(42)
    np.random.seed(42)
    timer.end_step()
    
    # Create output directory
    timer.start_step("Creating output directory")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    timer.end_step()
    
    # Check device
    timer.start_step("Checking device configuration")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  PyTorch is NOT using the GPU (CPU training will be slower)")
    timer.end_step()
    
    # Load and split data
    timer.start_step("Loading and splitting dataset")
    train_data, test_data = load_and_split_data(DATA_JSON_PATH, test_size=0.1, random_state=42)
    timer.end_step()
    
    # Initialize processor
    timer.start_step("Initializing processor")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ms", task="transcribe")
    print(f"✅ Processor loaded successfully")
    timer.end_step()
    
    # Initialize the data collator
    timer.start_step("Setting up data collator")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    timer.end_step()
    
    # Load the Whisper model
    timer.start_step("Loading Whisper model")
    try:
        # Try with 8-bit quantization first
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base", 
            quantization_config=quantization_config,
            device_map="auto"
        )
        print("✅ Model loaded with 8-bit quantization")
    except ImportError as e:
        print(f"8-bit quantization failed: {e}")
        print("Loading model without quantization (will use more memory)...")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base",
            device_map="auto"
        )
        print("✅ Model loaded without quantization")
    
    # Prepare the model for LoRA-compatible 8-bit training
    model = prepare_model_for_kbit_training(model)
    print(f"✅ Model prepared for k-bit training")
    timer.end_step()
    
    # Configure LoRA
    timer.start_step("Configuring LoRA")
    config = LoraConfig(
        r=16,  # Rank of LoRA decomposition
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention projections
        lora_dropout=0.1,  # Dropout applied to LoRA layers
        bias="none",  # Don't adapt bias terms
    )
    
    # Wrap the base model with LoRA
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  # Print which parameters are trainable
    print("✅ LoRA configuration applied")
    timer.end_step()
    
    # Evaluate baseline model performance
    timer.start_step("Evaluating baseline model")
    baseline_metrics = evaluate_baseline_model(model, processor, test_data, device)
    print(f"📊 Baseline WER: {baseline_metrics['wer']:.4f}")
    timer.end_step()
    
    # Define training hyperparameters
    timer.start_step("Setting up training arguments")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,  # Directory to save model checkpoints
        per_device_train_batch_size=1,  # Reduced batch size for memory
        gradient_accumulation_steps=8,  # Accumulate gradients for effective larger batch size
        learning_rate=5e-5,  # Lower learning rate for stable training
        warmup_steps=50,  # Warmup steps
        num_train_epochs=10,  # Maximum number of training epochs
        eval_strategy="steps",  # Evaluate every few steps
        logging_strategy="steps",  # Log every few steps
        logging_first_step=True,  # Log the very first training step
        logging_nan_inf_filter=True,  # Filter NaN/inf in logs
        eval_steps=25,  # Run evaluation every 25 steps
        report_to=[],  # No external logging
        fp16=True,  # Use mixed-precision (FP16) training
        per_device_eval_batch_size=1,  # Reduced batch size for evaluation
        generation_max_length=128,  # Max length for generation during eval
        logging_steps=5,  # Log every 5 steps
        remove_unused_columns=False,  # Needed for PEFT since forward signature is modified
        label_names=["labels"],  # Tells Trainer to pass labels explicitly
        dataloader_pin_memory=False,  # Disable pin memory for MPS
        dataloader_num_workers=0,  # Disable multiprocessing for MPS
        save_steps=25,  # Save checkpoints every 25 steps
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="eval_loss",  # Use eval loss for best model selection
        greater_is_better=False,  # Lower eval loss is better
    )
    print("✅ Training arguments configured")
    timer.end_step()
    
    # Create progress callback
    timer.start_step("Setting up progress tracking")
    progress_callback = TrainingProgressCallback()
    progress_callback.set_processor_and_dataset(processor, test_data)
    progress_callback.set_baseline_wer(baseline_metrics['wer'])
    print("✅ Progress tracking configured")
    timer.end_step()
    
    # Initialize Hugging Face Trainer
    timer.start_step("Setting up trainer")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=AudioTextDataset(data=train_data, processor=processor),
        eval_dataset=AudioTextDataset(data=test_data, processor=processor),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001),
            progress_callback
        ],
    )
    
    # Disable caching to avoid warnings during training
    model.config.use_cache = False
    print("✅ Trainer configured")
    timer.end_step()
    
    # Start training
    timer.start_step("TRAINING MODEL")
    print(f"🏋️  Starting training for {training_args.num_train_epochs} epochs...")
    print(f"📊 Total training samples: {len(train_data)}")
    print(f"📊 Total validation samples: {len(test_data)}")
    print(f"📊 Batch size: {training_args.per_device_train_batch_size}")
    print(f"📊 Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
    # Estimate training time
    samples_per_epoch = len(train_data)
    effective_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    batches_per_epoch = samples_per_epoch // effective_batch_size
    total_batches = batches_per_epoch * training_args.num_train_epochs
    
    print(f"📊 Effective batch size: {effective_batch_size}")
    print(f"📊 Estimated batches per epoch: {batches_per_epoch}")
    print(f"📊 Total estimated batches: {total_batches}")
    
    if torch.cuda.is_available():
        estimated_time_per_batch = 2  # seconds (rough estimate for GPU)
        estimated_total_time = (total_batches * estimated_time_per_batch) / 60  # minutes
        print(f"⏱️  Estimated training time: {estimated_total_time:.1f} minutes (GPU)")
    else:
        estimated_time_per_batch = 10  # seconds (rough estimate for CPU)
        estimated_total_time = (total_batches * estimated_time_per_batch) / 60  # minutes
        print(f"⏱️  Estimated training time: {estimated_total_time:.1f} minutes (CPU)")
    
    print("🚀 Training started...")
    trainer.train()
    timer.end_step()
    
    # Save final model
    timer.start_step("Saving final model")
    print("💾 Saving trained model...")
    trainer.save_model()
    print("💾 Saving processor...")
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Model and processor saved successfully!")
    timer.end_step()
    
    # Display training progress table
    timer.start_step("Displaying training results")
    progress_callback.display_progress_table()
    progress_callback.display_performance_metrics()
    timer.end_step()
    
    # Compute final WER after training
    timer.start_step("Computing final WER metrics")
    print("\n" + "="*50)
    print("COMPUTING FINAL WER METRICS")
    print("="*50)
    
    final_wer = compute_wer_metrics(model, processor, test_data, device)
    
    print(f"\nFINAL RESULTS:")
    print(f"Word Error Rate (WER): {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("="*50)
    timer.end_step()
    
    # Print final summary
    timer.print_summary()
    
    print("\n🎉 Training completed successfully!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
