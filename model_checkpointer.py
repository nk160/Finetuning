import os
import torch
from pathlib import Path

class ModelCheckpointer:
    """Handles model checkpointing with wandb integration"""
    def __init__(self, run):
        self.run = run
        self.best_der = float('inf')
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_metrics, is_best):
        """Save model checkpoint and optionally upload to wandb"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model if this is the best DER so far
        if is_best:
            self.best_der = val_metrics['der']
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Log to wandb
        self.run.save(str(latest_path))
        if is_best:
            self.run.save(str(best_path)) 