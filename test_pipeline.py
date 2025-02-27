import torch
from transformers import WhisperProcessor
from datasets import load_dataset
import os
from pathlib import Path
import torchaudio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from itertools import islice
import numpy as np

def test_pipeline():
    print("Testing pipeline components...")
    
    # Test CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test Whisper loading
    print("\nLoading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    print("Whisper processor loaded successfully")
    
    # Test data loading
    print("\nLoading test audio data...")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Take first 5 examples
    test_dataset = list(islice(dataset, 5))
    print(f"Test dataset loaded successfully with {len(test_dataset)} examples")
    
    # Test speaker embedding model
    print("\nLoading speaker embedding model...")
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        use_auth_token=os.getenv("HF_TOKEN")
    )
    print("Speaker embedding model loaded successfully")
    
    # Test processing one audio sample
    if test_dataset:
        print("\nTesting audio processing...")
        sample = test_dataset[0]
        audio = sample['audio']['array']
        print(f"Raw audio shape: {audio.shape}")
        
        # Reshape audio for speaker embedding (batch, channels, samples)
        audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0)
        print(f"Reshaped audio tensor shape: {audio_tensor.shape}")
        
        # Test Whisper processing
        inputs = processor(
            audio, 
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )
        print("Whisper processing successful")
        print(f"Whisper input features shape: {inputs.input_features.shape}")
        
        # Test speaker embedding
        embeddings = embedding_model(audio_tensor)
        print(f"Speaker embedding shape: {embeddings.shape}")
    
    print("\nAll components loaded successfully!")
    return True

if __name__ == "__main__":
    test_pipeline() 