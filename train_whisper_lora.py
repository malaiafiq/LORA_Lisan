from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback, TrainerCallback
import torch
import json
import librosa
import numpy as np
from typing import Dict, List, Any
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from sklearn.model_selection import train_test_split
import evaluate
from tqdm import tqdm
import os


def compute_baseline_wer(model, processor, eval_dataset, device, sample_size: int = 10) -> float:
    """Compute a baseline WER on a small subset before training, similar to the reference script."""
    wer_metric = evaluate.load("wer")
    model.eval()
    predictions: List[str] = []
    references: List[str] = []

    if len(eval_dataset) == 0:
        return 1.0

    size = min(sample_size, len(eval_dataset))
    indices = np.random.choice(len(eval_dataset), size, replace=False)

    with torch.no_grad():
        for idx in indices:
            item = eval_dataset[idx]
            audio_path = item['audio_path']
            reference_text = item.get('text', item.get('transcript', ''))
            if not reference_text:
                continue
            try:
                audio_array, _ = librosa.load(audio_path, sr=16000)
                input_features = processor.feature_extractor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_features.to(device)
                predicted_ids = model.generate(input_features, max_length=128)
                prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                predictions.append(prediction)
                references.append(reference_text)
            except Exception:
                continue

    if len(predictions) == 0 or len(references) == 0:
        return 1.0
    return wer_metric.compute(predictions=predictions, references=references)

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to track and display training progress in a formatted table."""
    
    def __init__(self):
        self.training_logs = []
        self.eval_logs = []
        self.wer_metric = evaluate.load("wer")
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Capture training and evaluation logs.

        Note: Hugging Face Trainer emits 'loss' for training steps (not 'train_loss').
        """
        if logs is not None:
            # Capture training loss from 'loss'
            if 'loss' in logs:
                self.training_logs.append({
                    'epoch': logs.get('epoch', 0),
                    'train_loss': logs.get('loss', 0),
                    'step': logs.get('step', 0)
                })
            # Capture evaluation metrics
            if 'eval_loss' in logs or 'eval_wer' in logs:
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
            sample_size = min(3, len(self.eval_dataset))
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
    
    def display_progress_table(self):
        """Display the training progress table."""
        print("\n" + "🏋️  TRAINING PROGRESS TABLE:")
        print("-" * 80)
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


def main():
    # Configuration - Change this to your JSON file path
    DATA_JSON_PATH = "D:\Afiq.hamidon\Lisan V2\organized_dataset.json"  # Update this path to your JSON file
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ms", task="transcribe")
    
    # Load and split the data (90:10 split)
    train_data, test_data = load_and_split_data(DATA_JSON_PATH, test_size=0.1, random_state=42)
    
    # Initialize the data collator to pad variable-length audio/text inputs
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Define training hyperparameters and settings
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints",  # Directory to save model checkpoints
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
        report_to=[],  # No external logging (removed wandb)
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

    # Load the Whisper model
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
        print("Model loaded with 8-bit quantization")
    except ImportError as e:
        print(f"8-bit quantization failed: {e}")
        print("Loading model without quantization (will use more memory)...")
        model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-base",
            device_map="auto"
        )

    # Prepare the model for LoRA-compatible 8-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
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

    # Create progress callback
    progress_callback = TrainingProgressCallback()
    
    # Initialize Hugging Face Trainer for training and evaluation with early stopping
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
    
    # Set processor and eval dataset for the progress callback
    progress_callback.set_processor_and_dataset(processor, test_data)

    # Disable caching to avoid warnings during training
    model.config.use_cache = False

    # Baseline evaluation before training (to match reference output)
    print("\n" + "="*50)
    print("BASELINE EVALUATION")
    print("="*50)
    device = next(model.parameters()).device
    baseline_wer = compute_baseline_wer(model, processor, test_data, device, sample_size=10)
    print(f"📊 Baseline WER: {baseline_wer:.4f}")

    # Start training
    print("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Display training progress table
    progress_callback.display_progress_table()
    
    # Compute final WER after training
    print("\n" + "="*50)
    print("COMPUTING FINAL WER METRICS")
    print("="*50)
    
    final_wer = compute_wer_metrics(model, processor, test_data, device)
    
    print(f"\nFINAL RESULTS:")
    print(f"Word Error Rate (WER): {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"Model saved to: {training_args.output_dir}")
    print("="*50)

    # Performance metrics summary (align with reference-style output)
    print("\n" + "="*80)
    print("🎯 TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"\n📊 PERFORMANCE METRICS:")
    print("-" * 40)
    wer_improvement = baseline_wer - final_wer
    print("Word Error Rate (WER):")
    print(f"  • Baseline:  {baseline_wer:.4f}")
    print(f"  • Final:     {final_wer:.4f}")
    print(f"  • Change:    {wer_improvement:+.4f} ({'✅ Improved' if wer_improvement > 0 else '❌ Degraded'})")


if __name__ == "__main__":
    main()