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
import requests
from tqdm import tqdm
import zipfile
import torchaudio
import soundfile as sf
from model_checkpointer import ModelCheckpointer
from torch.nn.utils import clip_grad_norm_

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

class VoxConverseStreamingDataset(Dataset):
    """Dataset for streaming VoxConverse data from Hugging Face"""
    def __init__(self, dataset, processor, max_duration=30, debug=False):
        self.dataset = dataset
        self.processor = processor
        self.max_duration = max_duration
        self.debug = debug
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        
        # Create speaker mapping
        self.speaker_to_idx = {}
        for item in dataset:
            for speaker in item['speakers']:
                if speaker not in self.speaker_to_idx:
                    self.speaker_to_idx[speaker] = len(self.speaker_to_idx)
        
        # Get total number of speakers from config
        self.num_speakers = wandb.config.num_speakers
        
        if debug:
            print(f"Initialized dataset with {len(dataset)} examples")
            print(f"Found {len(self.speaker_to_idx)} unique speakers")
            print(f"Total speakers in config: {self.num_speakers}")
            print(f"Speaker mapping: {self.speaker_to_idx}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            audio = item['audio']['array']
            
            # Process audio and get features
            inputs = self.processor(
                audio, 
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            encoder_output_length = inputs.input_features.shape[-1] // 2
            speaker_labels = torch.zeros(self.num_speakers, encoder_output_length)
            
            # Debug original values
            if self.debug and idx == 0:
                print("\nDebug Timestamp Conversion:")
                print(f"Audio length: {len(audio)} samples")
                print(f"Encoder length: {encoder_output_length} frames")
                print(f"Sample rate: 16000 Hz")
            
            # Convert timestamps using sample rate
            sample_rate = 16000  # VoxConverse uses 16kHz
            active_speakers = 0
            for start, end, speaker in zip(
                item['timestamps_start'],
                item['timestamps_end'],
                item['speakers']
            ):
                # Convert seconds to frames
                start_frame = max(0, int((start * sample_rate / 160) // 2))  # 160 samples per frame, divide by 2 for encoder
                end_frame = min(encoder_output_length, int((end * sample_rate / 160) // 2))
                
                if start_frame >= end_frame:
                    continue
                    
                speaker_idx = self.speaker_to_idx[speaker]
                if speaker_idx < self.num_speakers:
                    speaker_labels[speaker_idx, start_frame:end_frame] = 1
                    active_speakers += 1
                    
                    # Debug first few conversions
                    if self.debug and idx == 0 and active_speakers <= 3:
                        print(f"\nSpeaker {speaker} ({start:.2f}s - {end:.2f}s):")
                        print(f"Frames {start_frame} to {end_frame}")
            
            speaker_labels = speaker_labels.t()
            
            if self.debug and idx == 0:
                print(f"\nDebug Labels:")
                print(f"Active speakers found: {active_speakers}")
                print(f"Label sparsity: {speaker_labels.sum() / speaker_labels.numel():.4f}")
                print(f"Frame coverage: {(speaker_labels.sum(1) > 0).float().mean():.4f}")
            
            return {
                'input_features': inputs.input_features.squeeze(0),  # Remove batch dim
                'attention_mask': inputs.attention_mask.squeeze(0),  # Remove batch dim
                'speaker_labels': speaker_labels,  # [time/2, num_speakers]
                'speaker_embeddings': self.embedding_model(torch.tensor(audio).unsqueeze(0).unsqueeze(0))
            }
            
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            raise e

def download_voxconverse_data(debug=False):
    """Download VoxConverse dataset files"""
    # Create voxconverse directory
    voxconverse_path = Path("voxconverse")
    voxconverse_path.mkdir(exist_ok=True)
    
    # URLs for test data (smaller set for testing)
    test_files = {
        'audio': [
            'https://raw.githubusercontent.com/joonson/voxconverse/master/data/voxconverse/test/abfgy.wav',
            'https://raw.githubusercontent.com/joonson/voxconverse/master/data/voxconverse/test/abgih.wav'
        ],
        'rttm': [
            'https://raw.githubusercontent.com/joonson/voxconverse/master/data/voxconverse/test/abfgy.rttm',
            'https://raw.githubusercontent.com/joonson/voxconverse/master/data/voxconverse/test/abgih.rttm'
        ]
    }
    
    def download_file(url, dest_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            # Verify WAV files after download
            if dest_path.suffix == '.wav':
                try:
                    waveform, sample_rate = torchaudio.load(dest_path)
                    if debug:
                        print(f"Successfully loaded {dest_path.name}: {waveform.shape}, {sample_rate}Hz")
                except Exception as e:
                    print(f"Error verifying {dest_path}: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False
    
    # Download and verify files
    success = True
    for audio_url in test_files['audio']:
        filename = Path(audio_url).name
        dest_path = voxconverse_path / filename
        if not dest_path.exists() or not download_file(audio_url, dest_path):
            success = False
            
    for rttm_url in test_files['rttm']:
        filename = Path(rttm_url).name
        dest_path = voxconverse_path / filename
        if not dest_path.exists() or not download_file(rttm_url, dest_path):
            success = False
    
    if not success:
        raise RuntimeError("Failed to download or verify some files")
        
    if debug:
        print("\nDownloaded files:")
        print("WAV files:", list(voxconverse_path.glob("*.wav")))
        print("RTTM files:", list(voxconverse_path.glob("*.rttm")))
    
    return voxconverse_path

def prepare_dataset(processor, batch_size, debug=False):
    """Load and prepare VoxConverse dataset from Hugging Face"""
    try:
        if debug:
            print("Loading VoxConverse dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("diarizers-community/voxconverse")
        
        # Take smaller subsets for testing
        dev_subset = dataset['dev'].select(range(10))  # First 10 files
        test_subset = dataset['test'].select(range(5))  # First 5 files
        
        if debug:
            print(f"Using {len(dev_subset)} dev files and {len(test_subset)} test files")
        
        # Create dataset instances
        train_dataset = VoxConverseStreamingDataset(
            dev_subset,  # Use smaller dev set for training
            processor,
            max_duration=30,
            debug=debug
        )
        
        val_dataset = VoxConverseStreamingDataset(
            test_subset,  # Use smaller test set for validation
            processor,
            max_duration=30,
            debug=debug
        )
        
        if debug:
            print(f"Found {len(train_dataset)} training files")
            print(f"Found {len(val_dataset)} validation files")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Increase from 2
            pin_memory=True,
            prefetch_factor=4,  # Increase from 2
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
    pred_diarization: [batch, time, speakers] binary predictions
    true_diarization: [batch, time, speakers] binary ground truth
    """
    # Convert to float32 for stable computation
    pred = pred_diarization.astype(np.float32)
    true = true_diarization.astype(np.float32)
    
    # Compute frame-level errors
    false_alarm = np.sum((pred == 1) & (true == 0))
    missed_detection = np.sum((pred == 0) & (true == 1))
    
    # Total speech frames
    total_speech = np.sum(true)
    
    # DER calculation (clipped to [0, 1])
    der = np.clip((false_alarm + missed_detection) / max(total_speech, 1e-6), 0, 1)
    
    return der.item()

class DiarizationModel(torch.nn.Module):
    """Combined Whisper and Speaker Diarization Model"""
    def __init__(self, whisper_model, num_speakers=4):
        super().__init__()
        self.whisper = whisper_model
        hidden_size = self.whisper.config.d_model
        
        # Temporal context with residual connection
        self.temporal_context = ResidualConvBlock(hidden_size)
        
        # Multi-head self attention
        self.self_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # LSTM with skip connection
        self.sequence_layer = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )
        
        # Final classification with deeper network
        self.speaker_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_size // 2, num_speakers)
        )
        
        self.final_norm = torch.nn.BatchNorm1d(num_speakers)
        
        # Initialize weights properly
        for m in self.speaker_classifier.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
    def forward(self, input_features, attention_mask, speaker_embeddings=None):
        # Get Whisper encodings
        encoder_outputs = self.whisper.model.encoder(
            input_features,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Apply temporal convolutions with residual
        conv_input = hidden_states.transpose(1, 2)
        temporal_features = self.temporal_context(conv_input)
        temporal_features = temporal_features.transpose(1, 2)
        
        # Ensure attention mask is boolean and properly sized
        key_padding_mask = (~attention_mask[:, ::2].to(torch.bool))  # Invert and convert to bool
        
        # Apply self-attention
        attn_output, _ = self.self_attention(
            temporal_features, 
            temporal_features, 
            temporal_features,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection
        temporal_features = temporal_features + attn_output
        
        # Apply LSTM
        sequence_features, _ = self.sequence_layer(temporal_features)
        
        # Final classification
        speaker_logits = self.speaker_classifier(sequence_features)
        speaker_logits = speaker_logits.transpose(1, 2)
        speaker_logits = self.final_norm(speaker_logits)
        speaker_logits = speaker_logits.transpose(1, 2)
        
        return {
            'encoder_outputs': encoder_outputs,
            'speaker_logits': speaker_logits,
            'attention_weights': attn_output
        }

class ResidualConvBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = torch.nn.GroupNorm(8, channels)
        self.norm2 = torch.nn.GroupNorm(8, channels)
        self.dropout = torch.nn.Dropout(0.2)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        return x + residual

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler, run):
    """Train one epoch of the diarization model"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    accumulation_steps = 4
    max_grad_norm = 1.0  # Add gradient clipping threshold
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Move batch to device
            input_features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            speaker_labels = batch['speaker_labels'].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(input_features=input_features, attention_mask=attention_mask)
                
                # Base loss
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs['speaker_logits'],
                    speaker_labels.float(),
                    reduction='none'
                )
                
                # Add focal term
                probs = outputs['speaker_logits'].sigmoid()
                pt = torch.where(speaker_labels == 1, probs, 1 - probs)
                focal_weight = (1 - pt) ** 2
                
                # Compute loss components
                bce_loss = (bce * focal_weight).mean()
                sparsity_loss = 0.3 * probs.mean()
                l2_loss = 0.01 * sum(p.pow(2.0).sum() for p in model.parameters())
                
                # Combined loss
                loss = bce_loss + sparsity_loss + l2_loss
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Log gradient norm
                if batch_idx == 0:
                    run.log({'gradient_norm': grad_norm})
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item()
            })
            
            # Log to wandb
            run.log({
                'batch_loss': loss.item(),
                'speaker_balance': speaker_labels.float().mean().item()
            })
            
            # Debug loss
            if batch_idx == 0:
                print("\nDebug Loss:")
                print(f"Loss value: {loss.item()}")
                print(f"Loss requires grad: {loss.requires_grad}")
                print(f"Logits mean: {outputs['speaker_logits'].mean().item()}")
                print(f"Labels mean: {speaker_labels.float().mean().item()}\n")
            
            # Clear memory more frequently
            if batch_idx % 10 == 0:
                clear_memory()
            
            # Debug predictions
            speaker_preds = outputs['speaker_logits'].sigmoid() > 0.5
            if batch_idx == 0:  # First batch
                print("\nDebug Speaker Predictions:")
                print(f"Prediction shape: {speaker_preds.shape}")
                print(f"Number of active predictions: {speaker_preds.sum().item()}")
                print(f"Number of true speakers: {speaker_labels.sum().item()}")
                print(f"Sample predictions:\n{speaker_preds[0, :10, :5]}\n")  # First sequence, first 10 frames, first 5 speakers
            
            # Log detailed metrics
            if batch_idx == 0:
                run.log({
                    'bce_loss': bce_loss.item(),
                    'sparsity_loss': sparsity_loss.item(),
                    'l2_loss': l2_loss.item(),
                    'total_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                clear_memory()
                continue
            else:
                raise e
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device, processor):
    """Validate the diarization model"""
    model.eval()
    total_loss = 0
    total_der = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            try:
                # Move batch to device
                input_features = batch['input_features'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                speaker_labels = batch['speaker_labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_features=input_features,
                    attention_mask=attention_mask
                )
                
                # Compute loss
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    outputs['speaker_logits'],
                    speaker_labels.float()
                )
                
                # Get speaker predictions
                speaker_preds = (outputs['speaker_logits'].sigmoid() > 0.5).float()
                
                # Compute DER
                batch_der = compute_diarization_error_rate(
                    speaker_preds.cpu().numpy(),
                    speaker_labels.cpu().numpy()
                )
                
                total_loss += loss.item()
                total_der += batch_der
                num_batches += 1
                
                # Log detailed metrics
                metrics = {
                    'val_loss': loss.item(),
                    'der': batch_der,
                    'speaker_activation': speaker_preds.float().mean().item(),
                    'true_speaker_ratio': speaker_labels.float().mean().item()
                }
                
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
    
    # Calculate metrics
    metrics = {
        'val_loss': total_loss / num_batches,
        'der': total_der / num_batches
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

def setup_training(config):
    """Initialize model, optimizer, and other training components"""
    # Initialize Whisper model and processor
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Get number of speakers from dataset if not specified
    if config["num_speakers"] is None:
        dataset = load_dataset("diarizers-community/voxconverse")
        unique_speakers = set()
        
        # Check both dev and test sets
        for split in ['dev', 'test']:
            for item in dataset[split]:
                unique_speakers.update(item['speakers'])
        
        num_speakers = len(unique_speakers)
        print(f"Found {num_speakers} unique speakers in dataset")
        wandb.config.update({"num_speakers": num_speakers}, allow_val_change=True)
    
    # Create combined model
    model = DiarizationModel(
        whisper_model=whisper_model,
        num_speakers=wandb.config.num_speakers
    ).to(DEVICE)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
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

def create_synthetic_dataset(debug=False):
    """Create synthetic audio and RTTM files for testing"""
    # Create voxconverse directory
    voxconverse_path = Path("voxconverse")
    voxconverse_path.mkdir(exist_ok=True)
    
    # Create synthetic audio files
    for file_id in ['test1', 'test2']:
        # Generate 30 seconds of audio
        duration = 30
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create two different frequencies for two speakers
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        
        # Save audio file
        audio_path = voxconverse_path / f"{file_id}.wav"
        sf.write(audio_path, audio, sample_rate)
        
        # Create corresponding RTTM file
        rttm_path = voxconverse_path / f"{file_id}.rttm"
        with open(rttm_path, 'w') as f:
            # Speaker 1: 0-15s
            f.write(f"SPEAKER {file_id} 1 0.000 15.000 <NA> <NA> SPEAKER1 <NA> <NA>\n")
            # Speaker 2: 10-25s (overlapping)
            f.write(f"SPEAKER {file_id} 1 10.000 15.000 <NA> <NA> SPEAKER2 <NA> <NA>\n")
    
    if debug:
        print("\nCreated synthetic dataset:")
        print("WAV files:", list(voxconverse_path.glob("*.wav")))
        print("RTTM files:", list(voxconverse_path.glob("*.rttm")))
    
    return voxconverse_path

def main():
    """Main training routine"""
    # Add at start of training
    torch.backends.cudnn.benchmark = True
    
    # Initialize wandb with test config
    run = wandb.init(
        project="WhisperVox",
        name="voxconverse-test",
        config={
            "dataset": "voxconverse",
            "model_type": "tiny.en",
            "batch_size": 32,  # Increase from 16
            "learning_rate": 5e-5,  # Slightly increase learning rate to compensate
            "epochs": 1,
            "max_audio_length": 30,
            "sampling_rate": 16000,
            "num_speakers": None,  # Will be set from dataset
            "patience": 1,
            "debug": True,
            "diarization_weight": 1.0,
            "max_grad_norm": 1.0,
            "warmup_steps": 100,
            "val_every_n_steps": 500,
            "gradient_accumulation_steps": 4
        }
    )
    
    print("Initializing training...")
    print(f"Device: {DEVICE}")
    
    # Get model components
    model, processor, optimizer = setup_training(run.config)
    
    # Prepare data
    train_loader, val_loader = prepare_dataset(
        processor, 
        run.config.batch_size,
        debug=run.config.debug
    )
    
    # Create scheduler after we have train_loader
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=run.config.learning_rate,
        epochs=run.config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000
    )
    
    # Initialize training utilities
    scaler = torch.cuda.amp.GradScaler()
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
            scheduler=scheduler,
            device=DEVICE,
            scaler=scaler,
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
            'val_loss': val_metrics['val_loss']
        })
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation DER: {val_metrics['der']:.4f}")
        print(f"Validation Loss: {val_metrics['val_loss']:.4f}")
        
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
        early_stopping(val_metrics['val_loss'])
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