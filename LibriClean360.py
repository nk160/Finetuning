import torch
import whisper
import wandb
import optuna
from datasets import load_dataset
from jiwer import wer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import json

# W&B setup
wandb.init(
    project="whisper-fine-tuning",
    name="librispeech-clean-360",
    config={
        "dataset": "train-clean-360",
        "model_type": "tiny.en",
        "batch_size": 24,
        "learning_rate": 2e-5,
        "epochs": 5,
        "validation_steps": 200,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 3
    }
)

class LibriSpeechDataset(Dataset):
    """Custom Dataset for LibriSpeech"""
    def __init__(self, dataset, split="train"):
        self.dataset = dataset[split]
        self.processor = whisper.pad_or_trim
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item['audio']['array']
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)
        return {
            'input_features': mel,
            'labels': item['text']
        }

def prepare_dataset(batch_size):
    """Load and prepare LibriSpeech dataset"""
    dataset = load_dataset("librispeech_asr", "clean")
    train_dataset = LibriSpeechDataset(dataset, "train.360")  # Using train.360
    val_dataset = LibriSpeechDataset(dataset, "validation")
    
    num_workers = min(16, os.cpu_count())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

class ModelCheckpointer:
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_metrics, is_best=False):
        checkpoint = {
            # ... existing metadata ...
            'dataset_info': {
                'train_samples': 104014,  # Updated for clean-360
                'valid_samples': 2703,
                'test_samples': 2620
            },
            # ... rest remains the same ...
        } 

def objective(trial):
    # Define the hyperparameter search space
    config = {
        "dataset": "train-clean-360",
        "model_type": "tiny.en",
        "batch_size": trial.suggest_int("batch_size", 16, 48, step=8),  # Adjusted range
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "epochs": 5,
        "validation_steps": 200,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 4)
    }
    
    [Rest of objective function remains the same...] 