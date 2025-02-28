import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.optim.lr_scheduler import OneCycleLR
import gc, psutil
import time

# ---------- Simplified Dataset with Binary Speech Detection ----------
class VoxSimpleDataset(Dataset):
    def __init__(self, dataset, processor, num_speakers, binary_mode=True):
        self.dataset = dataset
        self.processor = processor
        self.num_speakers = num_speakers
        self.binary_mode = binary_mode  # If True, only detect speech/non-speech
        
        # Map speaker IDs
        speaker_ids = {}
        for item in dataset:
            for spk in item['speakers']:
                if spk not in speaker_ids:
                    speaker_ids[spk] = len(speaker_ids)
        self.speaker_ids = speaker_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item['audio']['array']
        inputs = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        
        # Debug label creation
        print(f"\nProcessing item {idx}")
        print(f"Audio length: {len(audio)}")
        print(f"Encoder frames: {inputs.input_features.shape[-1] // 2}")
        print(f"Timestamps: {list(zip(item['timestamps_start'], item['timestamps_end']))}")
        
        # Create empty labels
        enc_len = inputs.input_features.shape[-1] // 2
        if self.binary_mode:
            # Binary mode: single channel for any speech
            labels = torch.zeros(1, enc_len)
        else:
            # Multi-speaker mode: one channel per speaker
            labels = torch.zeros(self.num_speakers, enc_len)
        
        # Calculate sample to frame ratio
        audio_len_sec = len(audio) / 16000
        frames_per_sec = enc_len / audio_len_sec
        
        # Mark speech frames
        for s, e, spk in zip(item['timestamps_start'], item['timestamps_end'], item['speakers']):
            # Convert to frame indices
            start_frame = max(0, min(int(s * frames_per_sec), enc_len-1))
            end_frame = max(0, min(int(e * frames_per_sec), enc_len))
            
            if start_frame < end_frame:
                if self.binary_mode:
                    # In binary mode, mark any speech as 1 in the single channel
                    labels[0, start_frame:end_frame] = 1
                else:
                    # In multi-speaker mode, mark speech per speaker
                    if spk in self.speaker_ids:
                        spk_idx = self.speaker_ids[spk]
                        if spk_idx < self.num_speakers:
                            labels[spk_idx, start_frame:end_frame] = 1
        
        print(f"Total active frames: {torch.sum(labels)}")
        
        return {
            "input_features": inputs.input_features.squeeze(0),
            "attention_mask": torch.ones_like(inputs.input_features[0][0]).unsqueeze(0).squeeze(0),
            "labels": labels.t()  # shape: [time, num_speakers] or [time, 1]
        }

# ---------- Simple Binary Speech Detector Model ----------
class BinarySpeechDetector(torch.nn.Module):
    def __init__(self, whisper_model):
        super().__init__()
        self.whisper = whisper_model
        hidden_size = self.whisper.config.d_model
        
        # Simple but effective architecture for binary detection
        self.detector = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, 1)  # Single output for speech/non-speech
        )
    
    def forward(self, input_features, attention_mask):
        # Extract features from Whisper encoder
        encoder_out = self.whisper.model.encoder(
            input_features, 
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state
        
        # Make binary speech/non-speech decisions
        return self.detector(encoder_out)

# ---------- Multi-speaker Diarization Model ----------
class DiarizationModel(torch.nn.Module):
    def __init__(self, whisper_model, num_speakers, pretrained_detector=None):
        super().__init__()
        self.whisper = whisper_model
        hidden_size = self.whisper.config.d_model
        
        # Use pretrained detector if available
        self.use_pretrained = pretrained_detector is not None
        self.pretrained_detector = pretrained_detector
        
        # Stronger regularization and deeper network
        self.diarizer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),  # Increased dropout
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.LayerNorm(hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_size // 2, num_speakers)
        )
        
        # Initialize final layer with slightly stronger negative bias
        self.diarizer[-1].bias.data.fill_(-0.65)
    
    def forward(self, input_features, attention_mask):
        # Extract features from Whisper encoder
        encoder_out = self.whisper.model.encoder(
            input_features, 
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state
        
        if self.use_pretrained:
            # Get binary speech detection first
            speech_logits = self.pretrained_detector(input_features, attention_mask)
            speech_probs = torch.sigmoid(speech_logits)
            
            # Apply speaker classification where speech is detected
            speaker_logits = self.diarizer(encoder_out)
            
            # Combine: only assign speakers where speech is detected
            # Apply speech probabilities as a mask to speaker logits
            masked_speaker_logits = speaker_logits * speech_probs
            return masked_speaker_logits
        else:
            # Direct multi-speaker classification
            return self.diarizer(encoder_out)

# ---------- Training Utilities ----------
def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_binary_detector(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, run):
    """Train the binary speech detector"""
    print("Training binary speech detector...")
    scaler = torch.amp.GradScaler()
    best_f1 = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                logits = model(feats, mask)
                
                # Binary classification loss with high weight on positive class
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, 
                    pos_weight=torch.tensor([8.0]).to(device)  # High weight for speech
                )
            
            # Backward and optimize
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            train_steps += 1
            
            # Monitor predictions
            if train_steps % 10 == 0:
                with torch.no_grad():
                    preds = (logits.sigmoid() > 0.5).float()
                    accuracy = (preds == labels).float().mean()
                    
                    # Calculate F1 score
                    pred_pos = preds.sum().item()
                    true_pos = (preds * labels).sum().item()
                    precision = true_pos / max(pred_pos, 1e-10)
                    recall = true_pos / max(labels.sum().item(), 1e-10)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                    
                    run.log({
                        "binary_loss": loss.item(),
                        "binary_accuracy": accuracy.item(),
                        "binary_f1": f1,
                        "binary_precision": precision,
                        "binary_recall": recall
                    })
            
            # Clear memory
            clear_mem()
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        val_f1 = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                feats = batch["input_features"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].float().to(device)
                
                # Forward pass
                logits = model(feats, mask)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels, 
                    pos_weight=torch.tensor([8.0]).to(device)
                )
                
                # Calculate metrics
                preds = (logits.sigmoid() > 0.5).float()
                pred_pos = preds.sum().item()
                true_pos = (preds * labels).sum().item()
                precision = true_pos / max(pred_pos, 1e-10)
                recall = true_pos / max(labels.sum().item(), 1e-10)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                
                val_loss += loss.item()
                val_steps += 1
                val_f1 += f1
        
        val_loss /= val_steps
        val_f1 /= val_steps
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/train_steps:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}")
        run.log({
            "epoch": epoch+1,
            "binary_train_loss": train_loss/train_steps,
            "binary_val_loss": val_loss,
            "binary_val_f1": val_f1
        })
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_binary_detector.pt")
            print(f"Saved new best model with F1={best_f1:.4f}")
    
    # Load best model
    model.load_state_dict(torch.load("best_binary_detector.pt"))
    return model

def train_diarizer(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, run):
    """Train with improved parameters"""
    print("Training diarization model...")
    scaler = torch.amp.GradScaler()
    best_der = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            
            with torch.amp.autocast(device_type='cuda'):
                logits = model(feats, mask)
                
                # Reduce positive weight
                pos_weight = torch.ones(run.config.num_speakers, device=device) * 4.0
                
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels,
                    reduction='mean',
                    pos_weight=pos_weight
                )
                
                # Reduce L1 regularization back to original value
                l1_lambda = 0.005
                l1_reg = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_reg
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            if train_steps % 5 == 0:
                with torch.no_grad():
                    # Use same threshold as validation
                    preds = (logits.sigmoid() > 0.5).float()
                    run.log({
                        "train_batch_loss": loss.item(),
                        "train_pred_ratio": preds.mean().item(),
                        "train_label_ratio": labels.mean().item()
                    })
        
        # Validation
        model.eval()
        val_der = validate_diarizer(model, val_loader, device, run)
        
        # Log epoch metrics
        run.log({
            "epoch": epoch+1,
            "train_loss": train_loss / train_steps,
            "val_der": val_der
        })
        
        print(f"Epoch {epoch+1}, Loss: {train_loss/train_steps:.3f}, DER: {val_der:.3f}")
        
        # Save best model
        if val_der < best_der:
            best_der = val_der
            torch.save(model.state_dict(), "best_diarizer.pt")
            print(f"Saved best model with DER: {best_der:.3f}")
    
    return model

def validate_diarizer(model, val_loader, device, run):
    """Validate with improved post-processing"""
    model.eval()
    total_fa = 0
    total_miss = 0
    total_speech = 0
    
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            
            logits = model(feats, mask)
            
            # Middle threshold
            preds = (logits.sigmoid() > 0.53).cpu().numpy()
            
            # Apply temporal filtering
            preds = filter_predictions(preds, min_duration=2, max_gap=3)
            
            refs = labels.cpu().numpy()
            
            fa = np.sum((preds == 1) & (refs == 0))
            miss = np.sum((preds == 0) & (refs == 1))
            speech = np.sum(refs)
            
            total_fa += fa
            total_miss += miss
            total_speech += speech
            
            print(f"\nValidation Stats:")
            print(f"False Alarms: {fa}")
            print(f"Misses: {miss}")
            print(f"Total Speech: {speech}")
            print(f"Predictions Active: {np.sum(preds == 1)}")
    
    der = (total_fa + total_miss) / max(total_speech, 1e-6)
    return der

def filter_predictions(preds, min_duration=2, max_gap=3):
    """Filter predictions to remove short segments and fill small gaps"""
    filtered = preds.copy()
    batch_size, time_steps, num_speakers = filtered.shape
    
    for b in range(batch_size):
        for s in range(num_speakers):
            # Get binary prediction series
            pred_series = filtered[b, :, s]
            
            # Fill small gaps
            for i in range(1, len(pred_series)-1):
                if pred_series[i] == 0:
                    if pred_series[i-1] == 1 and pred_series[i+1] == 1:
                        gap_length = 1
                        j = i + 1
                        while j < len(pred_series) and pred_series[j] == 1:
                            j += 1
                        if gap_length <= max_gap:
                            pred_series[i:j] = 1
            
            # Remove short segments
            segments = np.where(np.diff(np.concatenate(([0], pred_series, [0]))))[0]
            for start, end in zip(segments[::2], segments[1::2]):
                if end - start < min_duration:
                    pred_series[start:end] = 0
            
            filtered[b, :, s] = pred_series
    
    return filtered

def main():
    # Start wandb run
    run = wandb.init(project="WhisperVox", name="two-stage-approach", config={
        "binary_epochs": 20,   # Double training time
        "diarizer_epochs": 20, # Double training time
        "batch_size": 16,     # Larger batches
        "binary_lr": 5e-5,
        "diarizer_lr": 2e-5,
        "num_speakers": None
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    
    # Load datasets (use full dataset)
    ds = load_dataset("diarizers-community/voxconverse")
    train_subset = ds['dev']  # Full dev set
    val_subset = ds['test']   # Full test set
    
    # Count speakers
    spk_set = set()
    for item in train_subset:
        for spk in item['speakers']:
            spk_set.add(spk)
    num_speakers = len(spk_set)
    print(f"Found {num_speakers} unique speakers")
    run.config.update({"num_speakers": num_speakers}, allow_val_change=True)
    
    # -------------------------------------------------------------------
    # Stage 1: Binary Speech Detection
    # -------------------------------------------------------------------
    print("\n=== STAGE 1: Binary Speech Detection ===\n")
    
    # Create binary datasets
    binary_train_data = VoxSimpleDataset(train_subset, processor, num_speakers, binary_mode=True)
    binary_val_data = VoxSimpleDataset(val_subset, processor, num_speakers, binary_mode=True)
    
    binary_train_loader = DataLoader(
        binary_train_data, 
        batch_size=run.config.batch_size, 
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,  # Increase prefetching
        pin_memory=True
    )
    
    binary_val_loader = DataLoader(
        binary_val_data, 
        batch_size=run.config.batch_size, 
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,  # Add prefetching
        pin_memory=True
    )
    
    # Create binary detector model
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    binary_model = BinarySpeechDetector(whisper_model).to(device)
    
    # Optimizer only for binary model (remove scheduler for testing)
    binary_optimizer = torch.optim.AdamW(
        binary_model.parameters(), 
        lr=run.config.binary_lr,
        weight_decay=0.01
    )
    
    # Use None for scheduler during testing
    binary_scheduler = None

    # Train binary model
    trained_binary_model = train_binary_detector(
        binary_model,
        binary_train_loader,
        binary_val_loader,
        binary_optimizer,
        binary_scheduler,
        device,
        run.config.binary_epochs,
        run
    )
    
    # -------------------------------------------------------------------
    # Stage 2: Multi-speaker Diarization
    # -------------------------------------------------------------------
    print("\n=== STAGE 2: Multi-speaker Diarization ===\n")
    
    # Create multi-speaker datasets
    diarizer_train_data = VoxSimpleDataset(train_subset, processor, num_speakers, binary_mode=False)
    diarizer_val_data = VoxSimpleDataset(val_subset, processor, num_speakers, binary_mode=False)
    
    diarizer_train_loader = DataLoader(
        diarizer_train_data, 
        batch_size=run.config.batch_size, 
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,  # Add prefetching
        pin_memory=True
    )
    
    diarizer_val_loader = DataLoader(
        diarizer_val_data, 
        batch_size=run.config.batch_size, 
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,  # Add prefetching
        pin_memory=True
    )
    
    # Create diarizer model with pretrained binary detector
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    diarizer_model = DiarizationModel(
        whisper_model, 
        num_speakers,
        pretrained_detector=trained_binary_model
    ).to(device)
    
    # Optimizer and scheduler
    diarizer_optimizer = torch.optim.AdamW(
        diarizer_model.parameters(), 
        lr=run.config.diarizer_lr,
        weight_decay=0.01
    )
    
    # Add warmup scheduler
    diarizer_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        diarizer_optimizer,
        max_lr=run.config.diarizer_lr,
        epochs=run.config.diarizer_epochs,
        steps_per_epoch=len(diarizer_train_loader),
        pct_start=0.3  # 30% warmup
    )
    
    # Train diarizer model
    trained_diarizer_model = train_diarizer(
        diarizer_model,
        diarizer_train_loader,
        diarizer_val_loader,
        diarizer_optimizer,
        diarizer_scheduler,
        device,
        run.config.diarizer_epochs,
        run
    )
    
    # Save final model
    torch.save(trained_diarizer_model.state_dict(), "final_diarizer.pt")
    
    # Finish wandb run
    run.finish()

if __name__ == "__main__":
    main()