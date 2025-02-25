# W&B setup
wandb.init(
    project="whisper-fine-tuning",
    name="librispeech-other-500",
    config={
        "dataset": "train-other-500",
        "model_type": "tiny.en",
        "batch_size": 16,  # Reduced further due to more complex data
        "learning_rate": 1e-5,  # Reduced for potentially noisier data
        "epochs": 5,
        "validation_steps": 300,  # Increased for larger dataset
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": 4  # Increased for larger dataset
    }
)

def prepare_dataset(batch_size):
    """Load and prepare LibriSpeech dataset"""
    dataset = load_dataset("librispeech_asr", "other")  # Changed to "other"
    train_dataset = LibriSpeechDataset(dataset, "train.500")
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
            'dataset_info': {
                'train_samples': 148688,  # Updated for other-500
                'valid_samples': 2864,    # Updated validation set size
                'test_samples': 2939      # Updated test set size
            },
        }

def objective(trial):
    # Define the hyperparameter search space
    config = {
        "dataset": "train-other-500",
        "model_type": "tiny.en",
        "batch_size": trial.suggest_int("batch_size", 8, 32, step=8),  # Adjusted range
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 2e-5, log=True),  # Lower range
        "epochs": 5,
        "validation_steps": 300,
        "max_audio_length": 30,
        "sampling_rate": 16000,
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 3, 6)
    }
    
    [Rest of file remains the same as LibriClean360.py...] 