import torch
import wandb
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import gc
import psutil
import torch.cuda.amp as amp

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
HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
DIARIZATION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/pyannote_diarization")

def cache_diarization_model():
    """Set up diarization pipeline using latest models"""
    try:
        # Create token directory if it doesn't exist
        os.makedirs(os.path.dirname(HF_TOKEN_PATH), exist_ok=True)
        
        # Get token from environment or file
        hf_token = os.getenv("HF_TOKEN")  # First try environment variable
        if not hf_token:
            try:
                with open(HF_TOKEN_PATH, 'r') as f:
                    hf_token = f.read().strip()
            except FileNotFoundError:
                raise ValueError("HuggingFace token not found. Please set HF_TOKEN environment variable or create token file")
            
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
    """Load and prepare VoxConverse dataset with memory-efficient streaming"""
    try:
        # Define base paths - adjust these to your actual paths
        voxconverse_path = Path("voxconverse")
        audio_path = voxconverse_path / "audio"
        rttm_path = voxconverse_path / "dev"  # or "test" for test set
        
        def load_voxconverse_file(audio_file):
            """Load single audio file and its corresponding RTTM"""
            try:
                # Load audio with torchaudio's streaming loader
                import torchaudio
                waveform, sample_rate = torchaudio.load(
                    audio_file,
                    channels_first=True,
                    format="wav",
                    normalize=True
                )
                
                # Get corresponding RTTM file
                rttm_file = rttm_path / f"{audio_file.stem}.rttm"
                speaker_segments = parse_rttm(rttm_file)
                
                return {
                    'audio': waveform,
                    'sample_rate': sample_rate,
                    'speaker_segments': speaker_segments
                }
            except Exception as e:
                print(f"Error loading file {audio_file}: {str(e)}")
                return None
        
        def parse_rttm(rttm_file):
            """Parse RTTM file to get speaker segments"""
            segments = []
            with open(rttm_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        speaker_id = parts[7]
                        segments.append({
                            'start': start_time,
                            'duration': duration,
                            'speaker': speaker_id
                        })
            return segments
        
        class VoxConverseStreamingDataset(Dataset):
            def __init__(self, audio_path, processor, max_duration=30):
                self.audio_files = list(audio_path.glob("*.wav"))
                self.processor = processor
                self.max_duration = max_duration
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    use_auth_token=os.getenv("HF_TOKEN")
                )
            
            def __len__(self):
                return len(self.audio_files)
            
            def __getitem__(self, idx):
                try:
                    data = load_voxconverse_file(self.audio_files[idx])
                    if data is None:
                        raise ValueError(f"Could not load file {self.audio_files[idx]}")
                    
                    # Process audio in chunks if needed
                    audio = data['audio']
                    if audio.shape[1] > self.max_duration * data['sample_rate']:
                        # Take first max_duration seconds
                        audio = audio[:, :self.max_duration * data['sample_rate']]
                    
                    # Process audio using Whisper processor
                    inputs = self.processor(
                        audio.squeeze().numpy(),
                        sampling_rate=data['sample_rate'],
                        return_tensors="pt",
                        return_attention_mask=True
                    )
                    
                    # Get speaker embeddings for the chunk
                    speaker_embeddings = self.embedding_model(audio.squeeze().numpy())
                    
                    return {
                        'input_features': inputs.input_features.squeeze(),
                        'attention_mask': inputs.attention_mask.squeeze(),
                        'speaker_embeddings': speaker_embeddings,
                        'speaker_segments': data['speaker_segments']
                    }
                    
                except Exception as e:
                    print(f"Error processing item {idx}: {str(e)}")
                    raise e
        
        # Create dataset instances with streaming
        train_dataset = VoxConverseStreamingDataset(
            audio_path / "dev",
            processor,
            max_duration=30
        )
        
        val_dataset = VoxConverseStreamingDataset(
            audio_path / "test",
            processor,
            max_duration=30
        )
        
        # Create data loaders with proper memory management
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,  # Reduced number of workers
            pin_memory=True,
            prefetch_factor=2,  # Reduce prefetching
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
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
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Log memory usage periodically
            if batch_idx % 10 == 0:
                log_memory_usage()
            
            # Move batch to device
            input_features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
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
                    labels=batch['labels']
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
            
            # Clear memory periodically
            if batch_idx % 50 == 0:
                clear_memory()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # Skip this batch
                continue
            else:
                raise e
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device, processor):
    """Validate the diarization model"""
    model.eval()
    log_memory_usage()  # Log initial memory state
    
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
    
    clear_memory()  # Clear memory after validation
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

def log_memory_usage():
    """Log current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**2
        gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**2
        print(f"GPU Memory: {gpu_memory:.2f}MB (Max: {gpu_memory_max:.2f}MB)")
    
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**2
    print(f"RAM Usage: {ram_usage:.2f}MB")

def clear_memory():
    """Clear unused memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    """Main training routine"""
    # Initialize wandb first with all configs
    run = wandb.init(
        project="WhisperVox",
        name="voxconverse-initial",
        config={
            "dataset": "voxconverse",
            "model_type": "tiny.en",
            "batch_size": 32,  # Increased from 24 given 20GB VRAM
            "learning_rate": 2e-5,
            "epochs": 5,
            "max_audio_length": 30,
            "sampling_rate": 16000,
            "diarization_weight": 0.3,
            "num_speakers": 2,
            "patience": 1,
            "warmup_steps": 500,
            "gradient_accumulation_steps": 2  # Add gradient accumulation
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