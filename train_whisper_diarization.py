import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from whisper_diarization import WhisperForDiarization
from voxconverse_dataset import VoxConverseDataset
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
import wandb
import os

def train(
    batch_size=8,
    num_epochs=10,
    learning_rate=1e-4,
    warmup_steps=1000,
    max_grad_norm=1.0,
    device="cuda",
    debug=False  # Add debug flag
):
    # Initialize wandb with debug mode if needed
    wandb.init(project="whisper-diarization", mode="disabled" if debug else "online")
    
    # Initialize model
    model = WhisperForDiarization.from_pretrained(
        "openai/whisper-small",
        num_speakers=2  # Start with binary classification per speaker
    ).to(device)
    
    # Create datasets and dataloaders
    train_dataset = VoxConverseDataset("voxconverse", split="dev")
    val_dataset = VoxConverseDataset("voxconverse", split="test")
    
    if debug:
        # Take only first few samples
        train_dataset.file_ids = train_dataset.file_ids[:2]
        val_dataset.file_ids = val_dataset.file_ids[:1]
        num_epochs = 2  # Reduce epochs for debug
        batch_size = 2  # Smaller batch size
        warmup_steps = 2  # Minimal warmup
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
            
            for batch in train_loader:
                # Move batch to device
                input_features = batch["audio"].to(device)
                diarization_labels = batch["diarization_labels"].to(device)
                
                # Forward pass
                outputs = model(
                    input_features=input_features,
                    diarization_labels=diarization_labels
                )
                
                loss = outputs["loss"]
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                
                # Log metrics
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                
                progress.update(train_task, advance=1)
            
            # Validation
            model.eval()
            val_losses = []
            val_task = progress.add_task("Validation", total=len(val_loader))
            
            with torch.no_grad():
                for batch in val_loader:
                    input_features = batch["audio"].to(device)
                    diarization_labels = batch["diarization_labels"].to(device)
                    
                    outputs = model(
                        input_features=input_features,
                        diarization_labels=diarization_labels
                    )
                    
                    val_losses.append(outputs["loss"].item())
                    progress.update(val_task, advance=1)
            
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