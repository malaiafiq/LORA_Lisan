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
import warnings

# Suppress BitsAndBytes dtype casting warning (informational)
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")

# Cross-OS path normalization
def _normalize_path_for_current_os(file_path: str) -> str:
    """Normalize dataset audio paths between Windows and POSIX/WSL.

    Examples:
      - On POSIX/WSL: 'D:\\dir\\file.mp3' -> '/mnt/d/dir/file.mp3'
      - On Windows: keep original
    """
    if not isinstance(file_path, str) or len(file_path) < 2:
        return file_path
    # If running on POSIX (Linux/WSL) and path looks like a Windows drive path
    if os.name != "nt":
        # Detect patterns like 'C:\\...' or 'D:\\...'
        if len(file_path) >= 3 and file_path[1] == ':' and (file_path[2] == '\\' or file_path[2] == '/'):
            drive_letter = file_path[0].lower()
            remainder = file_path[3:].replace('\\', '/').lstrip('/')
            return f"/mnt/{drive_letter}/{remainder}"
        # Also handle backslashes generally on POSIX
        return file_path.replace('\\', '/')
    # On Windows, return as-is
    return file_path


class TrainingProgressCallback(TrainerCallback):
    """Custom callback to track and display training progress in a formatted table."""
    
    def __init__(self):
        self.training_logs = []
        self.eval_logs = []
        self.wer_metric = evaluate.load("wer")
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Capture training and evaluation logs."""
        if logs is not None:
            if 'train_loss' in logs or 'loss' in logs:
                # Coerce epoch to int for cleaner display
                raw_epoch = logs.get('epoch', state.epoch if hasattr(state, 'epoch') else 0)
                epoch_int = int(raw_epoch) if raw_epoch is not None else 0
                self.training_logs.append({
                    'epoch': epoch_int,
                    'train_loss': logs.get('train_loss', logs.get('loss', 0)),
                    'step': logs.get('step', state.global_step if hasattr(state, 'global_step') else 0)
                })
            # Capture eval_loss when it's available in logs (from trainer's evaluation)
            if 'eval_loss' in logs:
                raw_epoch = logs.get('epoch', state.epoch if hasattr(state, 'epoch') else 0)
                epoch_int = int(raw_epoch) if raw_epoch is not None else 0
                # Find existing entry for this epoch or create new one
                existing_entry = None
                for i, log_entry in enumerate(self.eval_logs):
                    if log_entry['epoch'] == epoch_int:
                        existing_entry = i
                        break
                
                if existing_entry is not None:
                    # Update existing entry with eval_loss, but preserve existing eval_wer if it exists
                    existing_wer = self.eval_logs[existing_entry].get('eval_wer', 0.0)
                    print(f"DEBUG on_log: Found existing entry {existing_entry} for epoch {epoch_int}, existing_wer={existing_wer:.4f}, updating eval_loss={logs.get('eval_loss', 0):.4f}")
                    self.eval_logs[existing_entry]['eval_loss'] = logs.get('eval_loss', 0)
                    # Preserve existing WER - don't overwrite if it was already set
                    if 'eval_wer' not in self.eval_logs[existing_entry]:
                        self.eval_logs[existing_entry]['eval_wer'] = existing_wer if existing_wer > 0 else 0.0
                    print(f"DEBUG on_log: After update - eval_loss={self.eval_logs[existing_entry].get('eval_loss', 0):.4f}, eval_wer={self.eval_logs[existing_entry].get('eval_wer', 0):.4f}")
                else:
                    # Create new entry (WER will be added in on_evaluate)
                    print(f"DEBUG on_log: Creating NEW entry for epoch {epoch_int} with eval_loss={logs.get('eval_loss', 0):.4f}, eval_wer=0.0")
                    self.eval_logs.append({
                        'epoch': epoch_int,
                        'eval_loss': logs.get('eval_loss', 0),
                        'eval_wer': 0.0,  # Will be updated in on_evaluate
                        'step': logs.get('step', state.global_step if hasattr(state, 'global_step') else 0)
                    })
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Compute WER during evaluation and update eval metrics."""
        # on_evaluate receives metrics, not logs!
        metrics = kwargs.get('metrics', {})
        print(f"DEBUG on_evaluate: Called with model={model is not None}, processor={hasattr(self, 'processor')}, eval_dataset={hasattr(self, 'eval_dataset')}")
        print(f"DEBUG on_evaluate: metrics={metrics}, state.epoch={state.epoch if hasattr(state, 'epoch') else 'N/A'}")
        
        if model is not None and hasattr(self, 'processor') and hasattr(self, 'eval_dataset'):
            try:
                device = next(model.parameters()).device
                print(f"DEBUG on_evaluate: About to compute WER...")
                wer = self._compute_wer_during_eval(model, device)
                print(f"DEBUG on_evaluate: WER computed = {wer:.4f}")
                
                # Get epoch information from state (on_evaluate is called after evaluation)
                raw_epoch = state.epoch if hasattr(state, 'epoch') else 0
                epoch_int = int(raw_epoch) if raw_epoch is not None else 0
                
                # Get eval_loss from metrics (passed to on_evaluate)
                eval_loss = metrics.get('eval_loss', 0)
                
                print(f"DEBUG on_evaluate: epoch_int={epoch_int}, eval_loss={eval_loss:.4f}, wer={wer:.4f}")
                
                # Find existing entry for this epoch or create new one
                existing_entry = None
                for i, log_entry in enumerate(self.eval_logs):
                    if log_entry['epoch'] == epoch_int:
                        existing_entry = i
                        break
                
                if existing_entry is not None:
                    # Update existing entry with WER
                    print(f"DEBUG on_evaluate: Found existing entry {existing_entry} for epoch {epoch_int}, updating with wer={wer:.4f}")
                    self.eval_logs[existing_entry]['eval_wer'] = wer
                    # Update eval_loss if we have it in metrics and it's not already set or is 0
                    if eval_loss > 0 and self.eval_logs[existing_entry].get('eval_loss', 0) == 0:
                        self.eval_logs[existing_entry]['eval_loss'] = eval_loss
                    elif eval_loss > 0:
                        # Update with the value from metrics (it might be more recent)
                        self.eval_logs[existing_entry]['eval_loss'] = eval_loss
                    print(f"DEBUG on_evaluate: After update - eval_loss={self.eval_logs[existing_entry].get('eval_loss', 0):.4f}, eval_wer={self.eval_logs[existing_entry].get('eval_wer', 0):.4f}")
                else:
                    # Create new entry - use eval_loss from metrics or 0 if not available yet
                    # (it will be updated in on_log when trainer logs it)
                    print(f"DEBUG on_evaluate: Creating NEW entry for epoch {epoch_int} with eval_loss={eval_loss:.4f}, wer={wer:.4f}")
                    self.eval_logs.append({
                        'epoch': epoch_int,
                        'eval_loss': eval_loss if eval_loss > 0 else 0.0,
                        'eval_wer': wer,
                        'step': state.global_step if hasattr(state, 'global_step') else 0
                    })
                
                # Display progress table at each evaluation (i.e., each epoch with current settings)
                self.display_progress_table()
            except Exception as e:
                print(f"Error computing WER during evaluation: {e}")
                import traceback
                traceback.print_exc()
                # Still log eval_loss even if WER fails - use metrics and state
                raw_epoch = state.epoch if hasattr(state, 'epoch') else 0
                epoch_int = int(raw_epoch) if raw_epoch is not None else 0
                eval_loss = metrics.get('eval_loss', 0)
                
                # Find or create entry
                existing_entry = None
                for i, log_entry in enumerate(self.eval_logs):
                    if log_entry['epoch'] == epoch_int:
                        existing_entry = i
                        break
                
                if existing_entry is not None:
                    self.eval_logs[existing_entry]['eval_wer'] = 0.0
                    if eval_loss > 0:
                        self.eval_logs[existing_entry]['eval_loss'] = eval_loss
                else:
                    self.eval_logs.append({
                        'epoch': epoch_int,
                        'eval_loss': eval_loss if eval_loss > 0 else 0.0,
                        'eval_wer': 0.0,
                        'step': state.global_step if hasattr(state, 'global_step') else 0
                    })
    
    def _compute_wer_during_eval(self, model, device):
        """Compute WER during evaluation on the full validation set."""
        try:
            model.eval()
            predictions = []
            references = []
            
            # Process ALL validation samples (not just a sample)
            print(f"Computing WER on {len(self.eval_dataset)} validation samples...")
            
            with torch.no_grad():
                for idx in range(len(self.eval_dataset)):
                    item = self.eval_dataset[idx]
                    audio_path = _normalize_path_for_current_os(item['audio_path'])
                    reference_text = item.get('text', item.get('transcript', ''))
                    
                    if not reference_text:
                        print(f"Warning: No reference text for validation sample {idx}, skipping...")
                        continue
                    
                    if not os.path.exists(audio_path):
                        print(f"Warning: Audio file not found: {audio_path}, skipping...")
                        continue
                        
                    try:
                        # Load and process audio
                        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
                        input_features = self.processor.feature_extractor(
                            audio_array,
                            sampling_rate=16000,
                            return_tensors="pt"
                        ).input_features.to(device)
                        
                        # Generate prediction from model outputs
                        predicted_ids = model.generate(input_features, max_length=128)
                        
                        # Decode predictions into readable text
                        prediction = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                        
                        # Store decoded prediction and reference
                        predictions.append(prediction.strip())
                        references.append(reference_text.strip())
                        
                    except Exception as e:
                        print(f"Error processing validation sample {idx} ({audio_path}): {e}")
                        continue
            
            # Calculate average WER across all validation samples
            if len(predictions) > 0 and len(references) > 0:
                # WER = (Substitutions + Deletions + Insertions) / Total Words in Reference
                wer = self.wer_metric.compute(predictions=predictions, references=references)
                print(f"Validation WER computed: {wer:.4f} ({wer*100:.2f}%) from {len(predictions)} samples")
                return wer
            else:
                print(f"Warning: No valid predictions computed (got {len(predictions)} predictions, {len(references)} references)")
                return 0.0
        except Exception as e:
            print(f"Error in _compute_wer_during_eval: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            model.train()
    
    def set_processor_and_dataset(self, processor, eval_dataset):
        """Set processor and eval dataset for WER computation."""
        self.processor = processor
        self.eval_dataset = eval_dataset
    
    def display_progress_table(self):
        """Display the training progress table."""
        print("\n" + "TRAINING PROGRESS TABLE:")
        print("-" * 80)
        print(f"{'Epoch':<8} {'Training Loss':<15} {'Validation Loss':<15} {'Validation WER':<15}")
        print("-" * 80)
        
        # Debug: Print what's in eval_logs
       # print(f"DEBUG: eval_logs has {len(self.eval_logs)} entries")
       # for i, log in enumerate(self.eval_logs):
       #     print(f"  Entry {i}: epoch={log.get('epoch')}, eval_loss={log.get('eval_loss', 0):.4f}, eval_wer={log.get('eval_wer', 0):.4f}")
        
        # Group logs by epoch - process in reverse to get latest values if duplicates exist
        epoch_data = {}
        
        # First, add all training losses
        for log in self.training_logs:
            epoch = log['epoch']
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            epoch_data[epoch]['train_loss'] = log['train_loss']
        
        # Then, add eval metrics - process in reverse so latest entries overwrite
        # This ensures we get the final merged values
        for log in reversed(self.eval_logs):
            epoch = log['epoch']
            if epoch not in epoch_data:
                epoch_data[epoch] = {}
            
            eval_loss = log.get('eval_loss', 0.0)
            eval_wer = log.get('eval_wer', 0.0)
            
            # Always update if we have a non-zero value, or if current is not set/zero
            if eval_loss > 0:
                epoch_data[epoch]['eval_loss'] = eval_loss
            elif epoch_data[epoch].get('eval_loss', 0.0) == 0.0:
                epoch_data[epoch]['eval_loss'] = eval_loss
            
            if eval_wer > 0:
                epoch_data[epoch]['eval_wer'] = eval_wer
            elif epoch_data[epoch].get('eval_wer', 0.0) == 0.0:
                epoch_data[epoch]['eval_wer'] = eval_wer
        
        # Display progress for each epoch
        for epoch in sorted(epoch_data.keys()):
            data = epoch_data[epoch]
            train_loss = data.get('train_loss', 0.0)
            eval_loss = data.get('eval_loss', 0.0)
            eval_wer = data.get('eval_wer', 0.0)
            
            # Print epoch as an integer (1, 2, 3, ...)
            print(f"{int(epoch):<8} {train_loss:<15.4f} {eval_loss:<15.4f} {eval_wer:<15.4f}")
        
        # Display final results
        if epoch_data:
            final_epoch = max(epoch_data.keys())
            final_data = epoch_data[final_epoch]
            print("-" * 80)
            print(f"{'FINAL':<8} {final_data.get('train_loss', 0.0):<15.4f} {final_data.get('eval_loss', 0.0):<15.4f} {final_data.get('eval_wer', 0.0):<15.4f}")
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

        # Create attention mask for input_features based on actual feature lengths
        # Get the original lengths before padding
        feature_lengths = []
        for feat in features:
            feat_tensor = feat["input_features"]
            if isinstance(feat_tensor, torch.Tensor):
                feature_lengths.append(feat_tensor.shape[-1])
            elif isinstance(feat_tensor, list):
                feature_lengths.append(len(feat_tensor[0]) if feat_tensor else 0)
            else:
                # Fallback: use the padded batch shape
                feature_lengths.append(batch["input_features"].shape[-1])
        
        max_length = batch["input_features"].shape[-1]
        
        # Create attention mask: 1 for real features, 0 for padding
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        for i, length in enumerate(feature_lengths):
            if length > 0:
                attention_mask[i, :length] = 1
        
        batch["attention_mask"] = attention_mask

        # Pad text labels - use tokenizer __call__ method for fast tokenizers
        # Labels are already token IDs (list of ints), so we need to pad them
        labels = [feat["labels"] for feat in features]
        # For fast tokenizers with already tokenized sequences, use pad with proper format
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Replace padding token IDs with -100 so they are ignored in loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Optionally remove BOS token if present at the beginning
        removed_bos = False
        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]
            removed_bos = True
        
        # Also adjust decoder attention mask if we removed BOS token
        if removed_bos and "attention_mask" in labels_batch:
            batch["decoder_attention_mask"] = labels_batch["attention_mask"][:, 1:]
        elif "attention_mask" in labels_batch:
            batch["decoder_attention_mask"] = labels_batch["attention_mask"]

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
        audio_path = _normalize_path_for_current_os(item['audio_path'])
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Validate audio data to prevent NaN gradients
        if np.isnan(audio_array).any() or np.isinf(audio_array).any():
            raise ValueError(f"Invalid audio data (NaN/Inf) in {audio_path}")
        if len(audio_array) == 0:
            raise ValueError(f"Empty audio file: {audio_path}")
        
        # Process audio to get input features
        input_features = self.processor.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features[0]
        
        # Validate input_features to prevent NaN gradients
        if torch.isnan(input_features).any() or torch.isinf(input_features).any():
            raise ValueError(f"Invalid input_features (NaN/Inf) from {audio_path}")
        
        # Process text to get labels - handle both 'text' and 'transcript' keys
        # Use tokenizer __call__ method directly (faster for fast tokenizers)
        text = item.get('text', item.get('transcript', ''))
        if not text:
            raise ValueError(f"Neither 'text' nor 'transcript' key found in item: {item.keys()}")
        # Use __call__ method with padding=False since we'll pad in the collator
        tokenized = self.processor.tokenizer(text, padding=False, return_tensors=None)
        labels = tokenized["input_ids"]
        
        # Ensure labels are not empty
        if len(labels) == 0:
            raise ValueError(f"Empty labels for text: {text[:50]}...")
        
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
            audio_path = _normalize_path_for_current_os(item['audio_path'])
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
                
                # Generate prediction with same parameters as validation during training
                predicted_ids = model.generate(input_features, max_length=128)
                prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Strip whitespace to match validation WER computation
                predictions.append(prediction.strip())
                references.append(reference_text.strip())
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    # Calculate WER
    if len(predictions) > 0 and len(references) > 0:
        wer = wer_metric.compute(predictions=predictions, references=references)
        return wer
    else:
        return 1.0  # Return worst possible WER if no valid predictions


def make_compute_metrics(processor):
    """
    Create a compute_metrics function that has access to the processor.
    This computes WER and makes it available for metric_for_best_model.
    """
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(eval_pred):
        """
        Compute metrics for evaluation.
        This is called by the trainer during evaluation.
        """
        predictions, labels = eval_pred
        
        # Handle predictions - they might be tensors, numpy arrays, or lists
        if hasattr(predictions, 'cpu'):
            # Tensor - convert to numpy
            predictions = predictions.cpu().numpy()
        elif isinstance(predictions, list):
            # List of sequences - convert each to numpy array if needed
            # Handle variable-length sequences by padding or processing individually
            try:
                predictions = np.array(predictions, dtype=object)
            except (ValueError, TypeError):
                # If it's a ragged array, keep as list
                pass
        elif not isinstance(predictions, np.ndarray):
            try:
                predictions = np.array(predictions)
            except (ValueError, TypeError):
                # If conversion fails, keep as is
                pass
        
        # Handle labels similarly
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        elif isinstance(labels, list):
            try:
                labels = np.array(labels, dtype=object)
            except (ValueError, TypeError):
                pass
        elif not isinstance(labels, np.ndarray):
            try:
                labels = np.array(labels)
            except (ValueError, TypeError):
                pass
        
        # Process predictions and labels - handle both arrays and lists
        decoded_preds = []
        decoded_labels = []
        
        # Get batch size
        if isinstance(predictions, np.ndarray):
            batch_size = predictions.shape[0]
        elif isinstance(predictions, list):
            batch_size = len(predictions)
        else:
            batch_size = 1
        
        # Decode each sequence individually to handle variable lengths
        for i in range(batch_size):
            try:
                # Get prediction for this sample
                if isinstance(predictions, np.ndarray):
                    pred_seq = predictions[i]
                else:
                    pred_seq = predictions[i] if i < len(predictions) else []
                
                # Get label for this sample
                if isinstance(labels, np.ndarray):
                    label_seq = labels[i]
                else:
                    label_seq = labels[i] if i < len(labels) else []
                
                # Convert to list if needed
                if hasattr(pred_seq, 'tolist'):
                    pred_seq = pred_seq.tolist()
                if hasattr(label_seq, 'tolist'):
                    label_seq = label_seq.tolist()
                
                # Replace -100 in labels with pad_token_id
                if isinstance(label_seq, (list, np.ndarray)):
                    label_seq = [processor.tokenizer.pad_token_id if token == -100 else token for token in label_seq]
                
                # Decode
                pred_text = processor.decode(pred_seq, skip_special_tokens=True).strip()
                label_text = processor.decode(label_seq, skip_special_tokens=True).strip()
                
                # Only add non-empty pairs
                if pred_text and label_text:
                    decoded_preds.append(pred_text)
                    decoded_labels.append(label_text)
            except Exception as e:
                # Skip this sample if there's an error
                print(f"Warning: Error processing sample {i} in compute_metrics: {e}")
                continue
        
        # Compute WER if we have valid pairs
        if len(decoded_preds) > 0 and len(decoded_labels) > 0 and len(decoded_preds) == len(decoded_labels):
            try:
                wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
                return {"eval_wer": wer}
            except Exception as e:
                print(f"Warning: Error computing WER: {e}")
                return {"eval_wer": 1.0}
        else:
            return {"eval_wer": 1.0}  # Return worst WER if no valid pairs
    
    return compute_metrics


def main():
    # Accept dataset path via CLI and choose reasonable OS-specific default
    import argparse
    parser = argparse.ArgumentParser(description="Train Whisper LoRA")
    parser.add_argument(
        "--data_json",
        type=str,
        required=False,
        help="Path to organized_dataset.json",
    )
    args = parser.parse_args()

    if args.data_json:
        DATA_JSON_PATH = args.data_json
    else:
        if os.name == "nt":
            DATA_JSON_PATH = r"D:\Afiq.hamidon\Lisan V2\organized_dataset.json"
        else:
            DATA_JSON_PATH = "/mnt/d/Afiq.hamidon/Lisan V2/organized_dataset.json"
    
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
        learning_rate=1e-5,  # Reduced learning rate for stable training with 8-bit quantization
        warmup_steps=50,  # Warmup steps
        num_train_epochs=100,  # Maximum number of training epochs (set to 10 for testing)
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        logging_strategy="steps",  # Log every few steps
        logging_first_step=True,  # Log the very first training step
        logging_nan_inf_filter=True,  # Filter NaN/inf in logs
        report_to=[],  # No external logging (removed wandb)
        fp16=True,  # Use mixed-precision (FP16) training
        per_device_eval_batch_size=1,  # Reduced batch size for evaluation
        generation_max_length=128,  # Max length for generation during eval
        logging_steps=5,  # Log every 5 steps
        remove_unused_columns=False,  # Needed for PEFT since forward signature is modified
        label_names=["labels"],  # Tells Trainer to pass labels explicitly
        dataloader_pin_memory=False,  # Disable pin memory for MPS
        dataloader_num_workers=0,  # Disable multiprocessing for MPS
        save_strategy="epoch",  # Save checkpoints at the end of each epoch
        save_total_limit=3,  # Keep only 3 best checkpoints
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="eval_wer",  # Use WER for best model selection
        greater_is_better=False,  # Lower WER is better
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Use non-reentrant gradient checkpointing
        max_grad_norm=1.0,  # Gradient clipping to prevent NaN gradients
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
    
    # Enable gradient checkpointing with use_reentrant=False to avoid warnings
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Set use_reentrant=False for checkpointing
        if hasattr(model, "config"):
            model.config.use_cache = False  # Disable cache when using gradient checkpointing

    # Create progress callback
    progress_callback = TrainingProgressCallback()
    
    # Create compute_metrics function with access to processor
    compute_metrics_fn = make_compute_metrics(processor)
    
    # Initialize Hugging Face Trainer for training and evaluation with early stopping
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=AudioTextDataset(data=train_data, processor=processor),
        eval_dataset=AudioTextDataset(data=test_data, processor=processor),
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
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

    # Start training
    print("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Display training progress table
    progress_callback.display_progress_table()
    
    # Compute final WER after training
    print("\n" + "="*50)
    print("COMPUTING FINAL WER METRICS")
    print("="*50)
    
    device = next(model.parameters()).device
    final_wer = compute_wer_metrics(model, processor, test_data, device)
    
    print(f"\nFINAL RESULTS:")
    print(f"Word Error Rate (WER): {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"Model saved to: {training_args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()