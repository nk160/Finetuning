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
    name="librispeech-clean-100-lr2e-5",
    config={
        "dataset": "train-clean-100",
        "model_type": "tiny.en",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 5,
        "validation_steps": 100,
        "max_audio_length": 30,  # maximum audio length in seconds
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 2
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
        # Ensure audio is the right length
        audio = whisper.pad_or_trim(audio)
        # Convert to mel spectrogram
        mel = whisper.log_mel_spectrogram(audio)
        return {
            'input_features': mel,
            'labels': item['text']
        }

def prepare_dataset(batch_size):
    """Load and prepare LibriSpeech dataset"""
    dataset = load_dataset("librispeech_asr", "clean")
    train_dataset = LibriSpeechDataset(dataset, "train.100")
    val_dataset = LibriSpeechDataset(dataset, "validation")
    
    # Increase workers based on CPU cores
    num_workers = min(16, os.cpu_count())  # Use up to 16 workers
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Increased
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,  # Increased
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

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    try:
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        optimizer.zero_grad()
        for i, batch in enumerate(progress_bar):
            try:
                mel = batch['input_features'].to(device, non_blocking=True)
                labels = batch['labels']
                
                # Get tokenized labels
                tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
                target_ids = [tokenizer.encode(text) for text in labels]
                target_ids = torch.tensor(target_ids, device=device)
                
                # Forward pass
                with autocast():
                    output = model(mel)
                
                # Compute loss
                loss = criterion(output.transpose(1, 2), target_ids)
                
                loss = loss / wandb.config.gradient_accumulation_steps  # Scale loss
                loss.backward()
                
                if (i + 1) % wandb.config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
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
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            mel = batch['input_features'].to(device)
            
            # Get model predictions
            result = model.transcribe(mel)
            predictions = result["text"]
            
            all_predictions.extend(predictions)
            all_references.extend(batch['labels'])
    
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
    # Define the hyperparameter search space
    config = {
        "dataset": "train-clean-100",
        "model_type": "tiny.en",
        "batch_size": trial.suggest_int("batch_size", 16, 64, step=8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "epochs": 5,
        "validation_steps": 100,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 4)
    }
    
    # Initialize W&B for this trial
    wandb.init(
        project="whisper-fine-tuning-optuna",
        name=f"trial_{trial.number}",
        config=config,
        reinit=True
    )
    
    try:
        # Setup model and training
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("tiny.en").to(device)
        
        # Prepare datasets with trial-specific batch size
        train_loader, val_loader = prepare_dataset(config["batch_size"])
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        criterion = torch.nn.CrossEntropyLoss()
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=1, min_delta=0.001)
        
        best_wer = float('inf')
        
        # Training loop
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
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
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name="whisper-librispeech-optimization"
    )
    
    # Run optimization
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed
    
    # Print results
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

if __name__ == "__main__":
    main() 