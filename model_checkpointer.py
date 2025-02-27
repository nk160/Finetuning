import os
import torch
import wandb

class ModelCheckpointer:
    """Handles model checkpointing with wandb integration"""
    def __init__(self, run):
        self.run = run
        self.best_der = float('inf')
        self.checkpoint_dir = os.path.join(run.dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
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
        latest_path = os.path.join(self.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best model if needed
        if is_best:
            self.best_der = val_metrics['der']
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            
            # Log best model to wandb
            self.run.log_artifact(best_path, name='best_model', type='model')
            
        # Log latest model to wandb periodically
        if epoch % 5 == 0:
            self.run.log_artifact(latest_path, name=f'model_epoch_{epoch}', type='model') 