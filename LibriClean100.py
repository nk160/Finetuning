import torch
import wandb
import optuna
from datasets import load_dataset, DatasetDict
from jiwer import wer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from datasets import config

# Global model configuration
MODEL_NAME = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

# W&B setup
wandb.init(
    project="whisper-fine-tuning",
    name="librispeech-clean-100-test",
    config={
        "dataset": "train-clean-100",
        "model_type": "tiny.en",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 1,
        "validation_steps": 100,
        "max_audio_length": 30,  # maximum audio length in seconds
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 2
    }
)

class LibriSpeechDataset(Dataset):
    """Custom Dataset for LibriSpeech"""
    def __init__(self, dataset, processor, split="train"):
        self.dataset = dataset[split]
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item['audio']['array']
        # Process audio using Whisper processor
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        # Get text and process it
        labels = self.processor(
            text=item['text'],
            return_tensors="pt"
        ).input_ids
        
        return {
            'input_features': inputs.input_features.squeeze(),
            'labels': labels.squeeze()
        }

def prepare_dataset(dataset, processor, batch_size):
    """Load and prepare LibriSpeech dataset"""
    train_dataset = LibriSpeechDataset(dataset, processor, "train.clean.100")
    val_dataset = LibriSpeechDataset(dataset, processor, "validation.clean")
    
    # Increase workers based on CPU cores
    num_workers = min(16, os.cpu_count())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def compute_metrics(pred_texts, ref_texts):
    """Compute WER and additional metrics"""
    metrics = {
        "wer": wer(reference=ref_texts, hypothesis=pred_texts),
        "num_words_reference": sum(len(ref.split()) for ref in ref_texts),
        "num_words_hypothesis": sum(len(pred.split()) for pred in pred_texts),
        "avg_sequence_length": np.mean([len(ref.split()) for ref in ref_texts])
    }
    return metrics

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        wandb.log({
            "gpu_memory_allocated_gb": allocated,
            "gpu_memory_reserved_gb": reserved
        })

def train_epoch(model, train_loader, optimizer, device, scaler):
    try:
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            try:
                # Move input features to device
                input_features = batch['input_features'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast():
                    outputs = model(
                        input_features=input_features,
                        labels=labels
                    )
                
                # Get loss from model outputs
                loss = outputs.loss / wandb.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (i + 1) % wandb.config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log to W&B
                wandb.log({"batch_loss": loss.item()})
                
                if (i + 1) % 100 == 0:  # Log every 100 batches
                    log_gpu_memory()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM error in batch {i}. Skipping batch...")
                continue
        
        return total_loss / len(train_loader)
    except Exception as e:
        print(f"Error in training epoch: {str(e)}")
        raise e

def validate(model, val_loader, device):
    torch.cuda.empty_cache()  # Clear GPU cache before validation
    model.eval()
    total_wer = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                input_features=input_features,
                max_length=256,
                num_beams=5
            )
            
            # Decode predictions and references
            transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            references = processor.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(transcriptions)
            all_references.extend(references)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    return metrics

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelCheckpointer:
    def __init__(self, wandb_run):
        self.best_wer = float('inf')
        self.wandb_run = wandb_run
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_metrics, is_best=False):
        # Enhanced metadata for checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'model_config': {
                'model_type': wandb.config.model_type,
                'dataset': wandb.config.dataset,
                'batch_size': wandb.config.batch_size,
                'learning_rate': wandb.config.learning_rate
            },
            'best_wer': self.best_wer,
            'dataset_info': {
                'train_samples': 28539,  # from LibriSpeech clean-100 split
                'valid_samples': 2703,
                'test_samples': 2620
            },
            'timestamp': wandb.run.start_time,
            'run_id': wandb.run.id
        }
        
        # Save checkpoint locally and to W&B
        checkpoint_path = f"checkpoint_epoch_{epoch+1}"
        if is_best:
            checkpoint_path += "_best"
            
        torch.save(checkpoint, checkpoint_path + ".pt")
        
        # Create W&B artifact
        artifact_name = checkpoint_path
        model_artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Whisper model fine-tuned on LibriSpeech Clean 100 - Epoch {epoch+1}" + 
                       (" (Best)" if is_best else "")
        )
        
        # Add files to artifact
        model_artifact.add_file(checkpoint_path + ".pt")
        
        # Log artifact to W&B
        self.wandb_run.log_artifact(model_artifact)
        
    def check_and_save(self, model, optimizer, epoch, train_loss, val_metrics):
        current_wer = val_metrics["wer"]
        
        # Save regular checkpoint
        self.save_checkpoint(model, optimizer, epoch, train_loss, val_metrics)
        
        # Save best model if WER improved
        if current_wer < self.best_wer:
            self.best_wer = current_wer
            self.save_checkpoint(model, optimizer, epoch, train_loss, val_metrics, is_best=True)
            print(f"New best WER: {current_wer:.4f}")

def objective(trial):
    config = {
        "dataset": "train-clean-100",
        "model_type": "tiny.en",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 1,
        "validation_steps": 100,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 2
    }
    
    # Initialize W&B for this trial
    wandb.init(
        project="whisper-fine-tuning-optuna",
        name=f"trial_{trial.number}",
        config=config,
        reinit=True
    )
    
    try:
        # Setup model and training using HuggingFace implementation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "openai/whisper-tiny.en"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        
        # Set a longer timeout
        config.HF_DATASETS_TIMEOUT = 1000  # 1000 seconds

        # Try downloading with explicit cache directory
        dataset = load_dataset(
            "librispeech_asr",
            "clean",
            data_dir="./data/LibriSpeech/LibriSpeech",
            cache_dir="./data"
        )
        
        # Prepare datasets with trial-specific batch size
        train_loader, val_loader = prepare_dataset(dataset, processor, config["batch_size"])
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=1, min_delta=0.001)
        
        best_wer = float('inf')
        
        # Training loop
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = validate(model, val_loader, device)
            
            current_wer = val_metrics["wer"]
            trial.report(current_wer, epoch)
            
            # Update best WER
            if current_wer < best_wer:
                best_wer = current_wer
            
            # Early stopping check
            early_stopping(current_wer)
            if early_stopping.should_stop or trial.should_prune():
                break
        
        wandb.finish()
        return best_wer
    
    except Exception as e:
        wandb.finish()
        raise e

def main():
    # Initialize wandb first
    wandb.init(
        project="whisper-fine-tuning",
        name="librispeech-clean-100",
        config={
            "model_name": MODEL_NAME,
            "dataset": "train-clean-100",
            "batch_size": 8,
            "learning_rate": 1e-5,
            "max_steps": 4000,
            "warmup_steps": 500,
            "gradient_accumulation_steps": 2,
        }
    )

    # Setup model and training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    # Load dataset
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        data_dir="./data/LibriSpeech/LibriSpeech",
        cache_dir="./data"
    )
    train_loader, val_loader = prepare_dataset(dataset, processor, wandb.config.batch_size)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scaler = GradScaler()
    checkpointer = ModelCheckpointer(wandb.run)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="whisper-librispeech-optimization"
    )
    
    # Run optimization with 1 trial instead of 20
    study.optimize(objective, n_trials=1)
    
    # Print and save results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (WER): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    best_params = study.best_params
    with open("best_params.json", "w") as f:
        json.dump(best_params, f)
    
    wandb.finish()

if __name__ == "__main__":
    main() 