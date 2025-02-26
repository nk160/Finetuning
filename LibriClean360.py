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
import aiohttp

# Global model configuration
MODEL_NAME = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(MODEL_NAME)

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
    def __init__(self, dataset, processor, split="train"):
        self.dataset = dataset[split]
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            audio = item['audio']['array']
            
            if len(audio) == 0:
                raise ValueError("Empty audio file encountered")
            
            # Process inputs with attention mask
            inputs = self.processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Process text with max length constraint
            labels = self.processor.tokenizer(
                text=item['text'],
                return_tensors="pt",
                padding="max_length",
                max_length=448,
                truncation=True,
                return_attention_mask=True
            )
            
            return {
                'input_features': inputs.input_features.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': labels.input_ids.squeeze()
            }
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise e

def prepare_dataset(dataset, processor, batch_size):
    """Load and prepare LibriSpeech dataset"""
    train_dataset = LibriSpeechDataset(dataset, processor, "train.clean.360")  # Updated for 360h
    val_dataset = LibriSpeechDataset(dataset, processor, "validation.clean")
    
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

class ModelCheckpointer:
    def __init__(self, wandb_run):
        self.best_wer = float('inf')
        self.wandb_run = wandb_run
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_metrics, is_best=False):
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
                'train_samples': 104014,  # Updated for clean-360
                'valid_samples': 2703,
                'test_samples': 2620
            },
            'timestamp': wandb.run.start_time,
            'run_id': wandb.run.id
        }
        
        checkpoint_path = f"checkpoint_epoch_{epoch+1}"
        if is_best:
            checkpoint_path += "_best"
            
        torch.save(checkpoint, checkpoint_path + ".pt")
        
        model_artifact = wandb.Artifact(
            name=checkpoint_path,
            type="model",
            description=f"Whisper model fine-tuned on LibriSpeech Clean 360 - Epoch {epoch+1}" + 
                       (" (Best)" if is_best else "")
        )
        
        model_artifact.add_file(checkpoint_path + ".pt")
        self.wandb_run.log_artifact(model_artifact)
        
    def check_and_save(self, model, optimizer, epoch, train_loss, val_metrics):
        current_wer = val_metrics["wer"]
        
        self.save_checkpoint(model, optimizer, epoch, train_loss, val_metrics)
        
        if current_wer < self.best_wer:
            self.best_wer = current_wer
            self.save_checkpoint(model, optimizer, epoch, train_loss, val_metrics, is_best=True)
            print(f"New best WER: {current_wer:.4f}")

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
                
                if (i + 1) % 100 == 0:
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
    torch.cuda.empty_cache()
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

def objective(trial):
    config = {
        "dataset": "train-clean-360",
        "model_type": "tiny.en",
        "batch_size": trial.suggest_int("batch_size", 16, 48, step=8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "epochs": 5,
        "validation_steps": 200,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 4)
    }
    
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
        
        # Load dataset
        train_dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="train.360",
            trust_remote_code=True,
            storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
        )
        val_dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="validation",
            trust_remote_code=True,
            storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
        )
        train_loader, val_loader = prepare_dataset(train_dataset, processor, config["batch_size"])
        
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
            
            if current_wer < best_wer:
                best_wer = current_wer
            
            if early_stopping(current_wer) or trial.should_prune():
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
        name="librispeech-clean-360",
        config={
            "model_name": MODEL_NAME,
            "dataset": "train-clean-360",
            "batch_size": 24,
            "learning_rate": 2e-5,
            "max_steps": 4000,
            "warmup_steps": 500,
            "gradient_accumulation_steps": 3,
        }
    )

    # Setup model and training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    # Load dataset
    train_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.360",
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    val_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        trust_remote_code=True,
        storage_options={"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}},
    )
    train_loader, val_loader = prepare_dataset(train_dataset, processor, wandb.config.batch_size)
    
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
    
    # Run optimization
    study.optimize(objective, n_trials=20)
    
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