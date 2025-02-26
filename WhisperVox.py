import torch
import wandb
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Force numpy to load before pyannote
np.nan  # Pre-load numpy constants

# Then pyannote imports
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import aiohttp

# Global configurations
MODEL_NAME = "openai/whisper-tiny.en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIARIZATION_MODEL_PATH = "models/pyannote_diarization"  # Local path for cached model

def cache_diarization_model():
    """Set up diarization pipeline using latest models"""
    try:
        # Get token from environment or file
        token_path = "/root/.cache/huggingface/token"
        with open(token_path, 'r') as f:
            hf_token = f.read().strip()
            
        # Use the latest pipeline version
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to device without using global DEVICE
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device('cuda'))
        else:
            pipeline = pipeline.to(torch.device('cpu'))
        
        print("Diarization pipeline initialized")
        return pipeline
        
    except Exception as e:
        print(f"Error setting up diarization: {str(e)}")
        raise e

class VoxConverseDataset(Dataset):
    """Custom Dataset for VoxConverse diarization data"""
    def __init__(self, dataset, processor, split="train"):
        self.dataset = dataset[split]
        self.processor = processor
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb"
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            audio = item['audio']['array']
            
            if len(audio) == 0:
                raise ValueError("Empty audio file encountered")
            
            # Process audio using Whisper processor
            inputs = self.processor(
                audio, 
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Get speaker embeddings using pyannote
            speaker_embeddings = self.embedding_model(audio)
            
            # Process text with speaker information
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
                'labels': labels.input_ids.squeeze(),
                'speaker_embeddings': speaker_embeddings,
                'speaker_labels': torch.zeros(1)  # Placeholder for now
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise e 

def prepare_dataset(processor, batch_size):
    """Load and prepare Common Voice dataset"""
    try:
        # Load Common Voice dataset with streaming
        train_dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            split="train[:100]",  # Just 100 examples
            streaming=True,  # Stream instead of downloading all
        )
        
        val_dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            split="validation[:10]",  # Just 10 examples
            streaming=True,  # Stream instead of downloading all
        )
        
        # Create dataset instances
        train_data = VoxConverseDataset(train_dataset, processor, "train")
        val_data = VoxConverseDataset(val_dataset, processor, "validation")
        
        # Create data loaders
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(8, os.cpu_count()),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise e

def compute_diarization_error_rate(pred_diarization, true_diarization):
    """
    Compute Diarization Error Rate (DER)
    DER = (false_alarm + missed_detection + speaker_error) / total_speech_time
    """
    # TODO: Implement detailed DER calculation
    false_alarm = 0.0
    missed_detection = 0.0
    speaker_error = 0.0
    total_speech_time = 0.0
    
    der = (false_alarm + missed_detection + speaker_error) / max(total_speech_time, 1e-6)
    return der

class DiarizationModel(torch.nn.Module):
    """Combined Whisper and Speaker Diarization Model"""
    def __init__(self, whisper_model, num_speakers=2):
        super().__init__()
        self.whisper = whisper_model
        self.speaker_classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),  # 512 is Whisper's hidden size
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_speakers)
        )
        
    def forward(self, input_features, attention_mask, speaker_embeddings=None):
        # Get Whisper encodings
        encoder_outputs = self.whisper.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Add speaker classification
        speaker_logits = self.speaker_classifier(encoder_outputs.last_hidden_state)
        
        return {
            'encoder_outputs': encoder_outputs,
            'speaker_logits': speaker_logits
        }

class ModelCheckpointer:
    def __init__(self, run):
        self.best_der = float('inf')
        self.run = run
        
    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_metrics, is_best=False):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_metrics': val_metrics,
            'best_der': self.best_der
        }
        
        checkpoint_path = f"diarization_checkpoint_epoch_{epoch+1}"
        if is_best:
            checkpoint_path += "_best"
            
        torch.save(checkpoint, checkpoint_path + ".pt")
        
        # Log to wandb using run object
        artifact = self.run.Artifact(
            name=checkpoint_path,
            type="model",
            description=f"Whisper Diarization Model - Epoch {epoch+1}" + 
                       (" (Best)" if is_best else "")
        )
        artifact.add_file(checkpoint_path + ".pt")
        self.run.log_artifact(artifact)

def train_epoch(model, train_loader, optimizer, device, scaler, diarization_weight, run):
    """Train one epoch of the diarization model"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        try:
            # Move batch to device
            input_features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            speaker_embeddings = batch['speaker_embeddings'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    speaker_embeddings=speaker_embeddings
                )
                
                # Compute ASR loss
                asr_outputs = model.whisper(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    labels=labels
                )
                asr_loss = asr_outputs.loss
                
                # Compute diarization loss (speaker classification)
                diarization_loss = torch.nn.functional.cross_entropy(
                    outputs['speaker_logits'].view(-1, outputs['speaker_logits'].size(-1)),
                    batch['speaker_labels'].to(device).view(-1)
                )
                
                # Combined loss
                loss = asr_loss + diarization_weight * diarization_loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'asr_loss': asr_loss.item(),
                'diar_loss': diarization_loss.item()
            })
            
            # Log to wandb using run
            run.log({
                'batch_loss': loss.item(),
                'asr_loss': asr_loss.item(),
                'diarization_loss': diarization_loss.item()
            })
            
        except Exception as e:
            print(f"Error in training batch: {str(e)}")
            continue
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device, processor):
    """Validate the diarization model"""
    model.eval()
    total_der = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            try:
                # Move batch to device
                input_features = batch['input_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                speaker_embeddings = batch['speaker_embeddings'].to(device)
                
                # Forward pass
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    speaker_embeddings=speaker_embeddings
                )
                
                # Get speaker predictions
                speaker_preds = torch.argmax(outputs['speaker_logits'], dim=-1)
                
                # Compute DER
                batch_der = compute_diarization_error_rate(
                    speaker_preds.cpu().numpy(),
                    batch['speaker_labels'].numpy()
                )
                total_der += batch_der
                
                # Store predictions for WER calculation
                generated_ids = model.whisper.generate(
                    input_features=input_features,
                    attention_mask=attention_mask
                )
                
                transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
                references = processor.batch_decode(batch['labels'], skip_special_tokens=True)
                
                all_predictions.extend(transcriptions)
                all_references.extend(references)
                
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
    # Calculate metrics
    metrics = {
        'der': total_der / len(val_loader),
        'wer': compute_wer(all_predictions, all_references)
    }
    
    return metrics

def compute_wer(predictions, references):
    """Compute Word Error Rate"""
    from jiwer import wer
    return wer(references, predictions)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
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

def setup_training(config):  # Accept config as parameter
    """Initialize model, optimizer, and other training components"""
    # Initialize Whisper model and processor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Create combined model
    model = DiarizationModel(
        whisper_model=whisper_model,
        num_speakers=config["num_speakers"]  # Use passed config
    ).to(DEVICE)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],  # Use passed config
        weight_decay=0.01
    )
    
    return model, processor, optimizer

def main():
    """Main training routine"""
    # Initialize wandb first with all configs
    run = wandb.init(
        project="WhisperVox",
        name="voxconverse-initial",
        config={
            "dataset": "voxconverse",
            "model_type": "tiny.en",
            "batch_size": 24,
            "learning_rate": 2e-5,
            "epochs": 5,
            "max_audio_length": 30,
            "sampling_rate": 16000,
            "diarization_weight": 0.3,
            "num_speakers": 2,
            "patience": 1,
            "warmup_steps": 500
        }
    )
    
    print("Initializing training...")
    print(f"Device: {DEVICE}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Pass config to setup_training
    model, processor, optimizer = setup_training(run.config)
    
    # Prepare datasets
    train_loader, val_loader = prepare_dataset(processor, run.config.batch_size)
    
    # Initialize training utilities
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=run.config.patience,
        min_delta=0.001
    )
    checkpointer = ModelCheckpointer(run)
    
    # Training loop
    print("Starting training...")
    for epoch in range(run.config.epochs):
        print(f"\nEpoch {epoch+1}/{run.config.epochs}")
        
        # Training phase with added parameters
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            scaler=scaler,
            diarization_weight=run.config.diarization_weight,
            run=run  # Pass run object
        )
        
        # Validation phase with processor
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=DEVICE,
            processor=processor  # Pass processor
        )
        
        # Log metrics
        run.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_der': val_metrics['der'],
            'val_wer': val_metrics['wer']
        })
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation DER: {val_metrics['der']:.4f}")
        print(f"Validation WER: {val_metrics['wer']:.4f}")
        
        # Save checkpoint
        checkpointer.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_metrics=val_metrics,
            is_best=(val_metrics['der'] < checkpointer.best_der)
        )
        
        # Early stopping check
        early_stopping(val_metrics['der'])
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break
    
    print("Training completed!")
    run.finish()

if __name__ == "__main__":
    run = None
    try:
        main()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        if run is not None:
            run.finish()
        raise e 