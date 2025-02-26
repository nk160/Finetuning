import torch
import wandb
import optuna
import aiohttp
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
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 1,
        "validation_steps": 100,
        "max_audio_length": 30,  # maximum audio length in seconds
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 1
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
            
            # Process text
            labels = self.processor.tokenizer(
                text=item['text'],
                return_tensors="pt",
                padding="max_length",
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

def collate_fn(batch):
    """Custom collate function to handle variable length inputs"""
    # Filter out failed items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    input_features = []
    labels = []
    
    # Get max lengths
    max_input_length = max([item['input_features'].shape[-1] for item in batch])
    max_label_length = max([item['labels'].shape[-1] for item in batch])
    
    # Pad each item to max length
    for item in batch:
        input_pad = max_input_length - item['input_features'].shape[-1]
        label_pad = max_label_length - item['labels'].shape[-1]
        
        input_features.append(torch.nn.functional.pad(
            item['input_features'], 
            (0, input_pad), 
            'constant', 
            0
        ))
        labels.append(torch.nn.functional.pad(
            item['labels'], 
            (0, label_pad), 
            'constant', 
            -100  # padding token
        ))
    
    # Stack tensors
    input_features = torch.stack(input_features)
    labels = torch.stack(labels)
    
    return {
        'input_features': input_features,
        'labels': labels
    }

def prepare_dataset(dataset, processor, batch_size):
    """Load and prepare LibriSpeech dataset"""
    try:
        train_dataset = LibriSpeechDataset(dataset, processor, "train")
        val_dataset = LibriSpeechDataset(dataset, processor, "validation")
        
        num_workers = min(8, os.cpu_count() or 1)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        raise e

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
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
                
                if (i + 1) % 100 == 0:
                    log_gpu_memory()
                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"OOM error in batch {i}. Skipping batch...")
                continue
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                continue
        
        return total_loss / len(train_loader)
    except Exception as e:
        print(f"Error in training epoch: {str(e)}")
        raise e

def validate(model, val_loader, device):
    """Validation function for the model"""
    torch.cuda.empty_cache()  # Clear GPU cache before validation
    model.eval()
    total_wer = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            try:
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
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
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
    def __init__(self, run_name, output_dir="./checkpoints"):
        self.best_wer = float('inf')
        self.output_dir = output_dir
        self.run_name = run_name
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, model, processor, wer, epoch):
        """Save model if WER improves"""
        if wer < self.best_wer:
            self.best_wer = wer
            checkpoint_dir = os.path.join(self.output_dir, f"epoch_{epoch}_wer_{wer:.4f}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model and processor
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            
            # Log to wandb
            wandb.log({
                "best_wer": wer,
                "checkpoint_saved": checkpoint_dir
            })
            print(f"\nSaved checkpoint with WER: {wer:.4f} to {checkpoint_dir}")

def objective(trial):
    config = {
        "dataset": "train-clean-100",
        "model_type": "tiny.en",
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 1,
        "validation_steps": 100,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 1
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
        
        # Load dataset using exactly the provided approach
        train_dataset = load_dataset(
            "librispeech_asr",
            "clean",
            split="train.100",
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
        
        dataset = {"train": train_dataset, "validation": val_dataset}
        train_loader, val_loader = prepare_dataset(dataset, processor, config["batch_size"])
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scaler = GradScaler()
        early_stopping = EarlyStopping(patience=1, min_delta=0.001)
        checkpointer = ModelCheckpointer(wandb.run.name)
        
        best_wer = float('inf')
        
        # Training loop
        for epoch in range(config["epochs"]):
            train_loss = train_epoch(model, train_loader, optimizer, device, scaler)
            val_metrics = validate(model, val_loader, device)
            
            current_wer = val_metrics["wer"]
            trial.report(current_wer, epoch)
            
            # Save checkpoint if WER improved
            checkpointer.save_checkpoint(model, processor, current_wer, epoch)
            
            # Early stopping check
            early_stopping(current_wer)
            if early_stopping.should_stop or trial.should_prune():
                break
        
        wandb.finish()
        return best_wer
    
    except Exception as e:
        wandb.finish()
        raise e

def check_training_config():
    """Verify training configuration"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Training will be slow on CPU.")
    
    print("\nTraining Configuration:")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return True

def main():
    # Initialize wandb first
    wandb.init(
        project="whisper-fine-tuning",
        name="librispeech-clean-100",
        config={
            "model_name": MODEL_NAME,
            "dataset": "train-clean-100",
            "batch_size": 16,
            "learning_rate": 1e-5,
            "max_steps": 4000,
            "warmup_steps": 500,
            "gradient_accumulation_steps": 1,
        }
    )

    # Check training configuration
    if not check_training_config():
        raise RuntimeError("Training configuration check failed")

    # Setup model and training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    
    # Load dataset using exactly the provided approach
    train_dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="train.100",
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
    
    dataset = {"train": train_dataset, "validation": val_dataset}
    train_loader, val_loader = prepare_dataset(dataset, processor, wandb.config.batch_size)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scaler = GradScaler()
    checkpointer = ModelCheckpointer(wandb.run.name)
    
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