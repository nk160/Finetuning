import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from whisper_diarization import WhisperForDiarization, WhisperDiarizationConfig
from voxconverse_dataset import VoxConverseDataset
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import wandb
import os
import torchaudio

# Add at the beginning of the file, before any other torch operations
mp.set_start_method('spawn', force=True)

def train(
    batch_size=8,
    num_epochs=10,
    learning_rate=5e-5,  # Lower learning rate
    warmup_steps=1000,
    max_grad_norm=0.5,   # Lower gradient clipping
    device="cuda",
    debug=False  # Add debug flag
):
    # Initialize wandb with debug mode if needed
    wandb.init(project="whisper-diarization", mode="disabled" if debug else "online")
    
    # Initialize model with config
    config = WhisperDiarizationConfig.from_pretrained(
        "openai/whisper-small",
        num_speakers=4  # Match max_speakers from dataset
    )
    model = WhisperForDiarization(config).to(device)
    
    # Create datasets and dataloaders
    train_dataset = VoxConverseDataset("voxconverse", split="dev", debug=debug)
    val_dataset = VoxConverseDataset("voxconverse", split="dev", debug=debug)
    
    if debug:
        all_files = train_dataset.file_ids.copy()
        if len(all_files) < 2:
            raise ValueError(f"Not enough files in dataset (found {len(all_files)}, need at least 2)")
        
        # Check files for debugging
        print("\nChecking available files:")
        multi_speaker_files = []
        for file_id in all_files:  # Check all files since we have few
            rttm_path = train_dataset.rttm_files[file_id]
            segments = train_dataset.read_rttm(rttm_path)
            speakers = set(spk for _, _, spk in segments)
            print(f"\nFile {file_id}:")
            print(f"Number of speakers: {len(speakers)}")
            print(f"Speakers: {speakers}")
            if len(speakers) > 1:
                multi_speaker_files.append(file_id)
        
        if multi_speaker_files:
            print("\nFound multi-speaker files:", multi_speaker_files)
            # Use multi-speaker file for training if available
            train_dataset.file_ids = multi_speaker_files[:1]
            val_dataset.file_ids = multi_speaker_files[1:2] if len(multi_speaker_files) > 1 else [f for f in all_files if f not in multi_speaker_files][:1]
        else:
            print("\nWARNING: No multi-speaker files found")
            # Use available files
            train_dataset.file_ids = all_files[:1]
            val_dataset.file_ids = all_files[1:2]
        
        print(f"\nNumber of training files: {len(train_dataset.file_ids)}")
        print(f"Number of validation files: {len(val_dataset.file_ids)}")
        
        num_epochs = 2
        batch_size = 2
        warmup_steps = 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if debug else 4,  # No multiprocessing in debug mode
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if debug else 4,  # No multiprocessing in debug mode
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01  # Add weight decay
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_task = progress.add_task(f"Epoch {epoch+1}", total=len(train_loader))
            train_losses = []
            optimizer.zero_grad()  # Zero gradients at start
            
            accumulation_steps = 4  # Accumulate gradients
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_features = batch["input_features"].to(device)
                diarization_labels = batch["diarization_labels"].to(device)
                
                # Forward pass
                outputs = model(
                    input_features=input_features,
                    diarization_labels=diarization_labels
                )
                
                # Scale loss and append to train_losses
                loss = outputs["loss"]
                train_losses.append(loss.item())  # Store full loss
                
                # Scale for accumulation and backward
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Log metrics
                wandb.log({
                    "train_loss": loss.item(),  # Log full loss
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                
                progress.update(train_task, advance=1)
            
            # Print epoch stats
            train_loss = sum(train_losses) / len(train_losses)
            progress.console.print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            # Validation
            model.eval()
            val_losses = []
            val_task = progress.add_task("Validation", total=len(val_loader))
            
            with torch.no_grad():
                for batch in val_loader:
                    input_features = batch["input_features"].to(device)
                    diarization_labels = batch["diarization_labels"].to(device)
                    
                    outputs = model(
                        input_features=input_features,
                        diarization_labels=diarization_labels
                    )
                    
                    val_losses.append(outputs["loss"].item())
                    progress.update(val_task, advance=1)
                    
                    if debug:
                        # Print validation predictions
                        probs = torch.sigmoid(outputs["diarization_logits"])
                        print(f"\nValidation predictions:")
                        print(f"Mean probabilities per speaker: {probs.mean(dim=1).mean(dim=0)}")
                        print(f"Max probabilities: {probs.max().item():.4f}")
            
            if len(val_losses) > 0:  # Only compute val_loss if we have validation data
                val_loss = sum(val_losses) / len(val_losses)
                wandb.log({"val_loss": val_loss})
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained("best_whisper_diarization")
                    
                progress.console.print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    # Run in debug mode
    train(debug=True) 