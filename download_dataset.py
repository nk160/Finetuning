from datasets import load_dataset, config
import os

# Create data directory if it doesn't exist
os.makedirs('./data', exist_ok=True)

# Set even longer timeout
config.HF_DATASETS_TIMEOUT = 3000  # 3000 seconds

# Download only train-clean-100 subset
print("Downloading LibriSpeech train-clean-100 dataset...")
try:
    dataset = load_dataset(
        'librispeech_asr',
        'clean',
        split=['train.clean.100', 'validation.clean'],
        cache_dir='./data',
        num_proc=4  # Use multiple processes for downloading
    )
    print("Download complete!")
except Exception as e:
    print(f"Error during download: {str(e)}")
    print("Try downloading with different parameters...")
    try:
        dataset = load_dataset(
            'librispeech_asr',
            'clean',
            split=['train.clean.100', 'validation.clean'],
            cache_dir='./data',
            streaming=True  # Try streaming mode
        )
        print("Streaming download complete!")
    except Exception as e:
        print(f"Second attempt failed: {str(e)}")