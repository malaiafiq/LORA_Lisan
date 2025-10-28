from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import librosa
import json
from sklearn.model_selection import train_test_split
import evaluate

def load_and_split_data(json_path: str, test_size: float = 0.1, random_state: int = 42):
    """Load data from JSON file and split into train/test sets."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return train_data, test_data

def test_base_model():
    # Load the base model (no LoRA)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base", language="ms", task="transcribe")
    
    # Load test data
    train_data, test_data = load_and_split_data("/Volumes/Lisan/organized_dataset.json", test_size=0.1, random_state=42)
    
    print(f"Testing base Whisper model on {len(test_data)} test samples...")
    
    # Load WER metric
    wer_metric = evaluate.load("wer")
    
    predictions = []
    references = []
    
    model.eval()
    
    for i, item in enumerate(test_data):
        print(f"Processing {i+1}/{len(test_data)}: {item['audio_path']}")
        
        # Load and process audio
        audio_array, sample_rate = librosa.load(item['audio_path'], sr=16000)
        
        # Get transcription
        input_features = processor.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features
        
        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode prediction
        prediction = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Get reference text
        reference = item.get('text', item.get('transcript', ''))
        
        predictions.append(prediction)
        references.append(reference)
        
        print(f"Reference: {reference}")
        print(f"Prediction: {prediction}")
        print("-" * 50)
    
    # Calculate WER
    wer = wer_metric.compute(predictions=predictions, references=references)
    
    print(f"\n{'='*50}")
    print(f"BASE MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"Test samples: {len(test_data)}")
    print(f"{'='*50}")
    
    # Show some examples
    print(f"\nSAMPLE PREDICTIONS:")
    for i in range(min(5, len(predictions))):
        print(f"\nExample {i+1}:")
        print(f"Reference:  {references[i]}")
        print(f"Prediction: {predictions[i]}")

if __name__ == "__main__":
    test_base_model()
