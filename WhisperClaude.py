# Your code will go here

import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from torch.optim.lr_scheduler import OneCycleLR
import gc, psutil

# ---------- Simplified Dataset ----------
class VoxSimpleDataset(Dataset):
    def __init__(self, dataset, processor, num_speakers):
        self.dataset = dataset
        self.processor = processor
        self.num_speakers = num_speakers
        
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
        
        # Prepare label frames
        enc_len = inputs.input_features.shape[-1] // 2
        labels = torch.zeros(self.num_speakers, enc_len)
        for s, e, spk in zip(item['timestamps_start'], item['timestamps_end'], item['speakers']):
            if spk not in self.speaker_ids: 
                continue
            spk_idx = self.speaker_ids[spk]
            # Clip to audio length in seconds
            audio_len_sec = len(audio) / 16000  # 16kHz audio
            s = min(s, audio_len_sec)
            e = min(e, audio_len_sec)
            start_f = int(s * 50)
            end_f = int(e * 50)
            if 0 <= start_f < end_f <= enc_len:
                labels[spk_idx, start_f:end_f] = 1
        print(f"Total active frames: {torch.sum(labels)}")
        return {
            "input_features": inputs.input_features.squeeze(0),
            "attention_mask": torch.ones_like(inputs.input_features[0][0]).unsqueeze(0).squeeze(0),
            "labels": labels.t()  # shape: [time, num_speakers]
        }

# ---------- Model with Temporal Integration ----------
class DiarizationModel(torch.nn.Module):
    def __init__(self, whisper_model, num_speakers):
        super().__init__()
        self.whisper = whisper_model
        hidden = self.whisper.config.d_model
        
        # Improved convolutional block with residual connections
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.GroupNorm(8, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.GroupNorm(8, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2)
        )
        
        # Temporal self-attention with more heads
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden, 
            num_heads=8,  # Increased from 4
            dropout=0.1,
            batch_first=True
        )
        
        # Properly connect LSTM that was defined but not used in original
        self.temporal = torch.nn.LSTM(
            hidden, hidden//2, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Enhanced classifier with deeper network and stronger regularization
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.LayerNorm(hidden//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden//2, num_speakers)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    # Initialize bias to negative value for sparse predictions
                    torch.nn.init.constant_(m.bias, -0.2)
            elif isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input_features, attention_mask):
        # Get encoder outputs from Whisper
        enc_out = self.whisper.model.encoder(
            input_features, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        
        # Apply convolutional block with residual connection
        conv_in = enc_out.transpose(1, 2)
        conv_out = self.conv_block(conv_in)
        conv_out = conv_out + conv_in  # Residual connection
        x = conv_out.transpose(1, 2)
        
        # Apply self-attention with proper masking
        key_pad = (~attention_mask[:, ::2].bool())  # half frames
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_pad)
        x = x + attn_out  # Residual connection
        
        # Apply LSTM layer (was missing in original forward)
        x, _ = self.temporal(x)
        
        # Apply temporal smoothing before classification
        # This helps reduce isolated predictions
        x = self.apply_temporal_smoothing(x)
        
        # Final classification
        logits = self.classifier(x)
        
        # Apply negative bias to encourage sparse predictions
        return logits - 0.2  
    
    def apply_temporal_smoothing(self, x, kernel_size=3):
        """Apply temporal smoothing to reduce isolated predictions"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Use average pooling for smoothing
        # Pad to maintain sequence length
        padding = (kernel_size - 1) // 2
        x_padded = torch.nn.functional.pad(x.transpose(1, 2), (padding, padding), mode='replicate')
        x_smoothed = torch.nn.functional.avg_pool1d(x_padded, kernel_size, stride=1)
        
        return x_smoothed.transpose(1, 2)

# ---------- Training ----------
def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, run):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.75)
    for i, batch in enumerate(tqdm(loader, desc="Train")):
        feats = batch["input_features"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)
        
        with torch.amp.autocast(device_type='cuda'):
            logits = model(feats, mask)
            probs = logits.sigmoid()
            
            # Compute focal loss with stronger focus on hard negatives
            pt = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** 3  # Stronger focus on hard examples
            
            # Compute BCE loss with higher positive weight to address class imbalance
            # But also a penalty for false positives to reduce overprediction
            pos_weight = torch.ones(run.config.num_speakers, device=device) * 2.0  # Reduced from 3.0
            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, 
                labels, 
                reduction='none',
                pos_weight=pos_weight
            )
            
            # Apply focal weighting
            weighted_loss = loss_bce * focal_weight
            
            # Add false positive penalty term
            # This additional term specifically penalizes false positives more heavily
            fp_mask = (labels == 0) & (probs > 0.3)  # Identify likely false positives
            fp_penalty = 1.5 * torch.sum(probs * fp_mask.float()) / (torch.sum(fp_mask.float()) + 1e-6)
            
            # Add sparsity regularization to encourage fewer active predictions
            sparsity_loss = 0.1 * torch.mean(probs)
            
            # Final loss combines all components
            loss = weighted_loss.mean() + fp_penalty + sparsity_loss
            
        scaler.scale(loss / ACCUMULATION_STEPS).backward()
        total_loss += loss.item()
        
        # Log detailed loss components
        if i % 10 == 0:
            run.log({
                "batch_loss": loss.item(),
                "fp_penalty": fp_penalty.item(),
                "sparsity_loss": sparsity_loss.item(),
                "bce_loss": weighted_loss.mean().item(),
                "positive_ratio": labels.float().mean().item(),
                "pred_ratio": (probs > 0.5).float().mean().item()
            })
            
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Tighter gradient clipping
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        if i % 5 == 0:  # Clear cache more frequently
            torch.cuda.empty_cache()
        clear_mem()
        
    return total_loss / len(loader)

def post_process_predictions(preds, min_duration=2):
    """
    Apply post-processing to remove short segments and noise
    preds: numpy array of shape [batch, time, speakers] with binary values
    min_duration: minimum number of consecutive frames to keep (removes shorter segments)
    """
    processed = preds.copy()
    batch_size, time_steps, num_speakers = processed.shape
    
    for b in range(batch_size):
        for s in range(num_speakers):
            # Get binary prediction series for this speaker
            pred_series = processed[b, :, s]
            
            # Find consecutive segments
            changes = np.diff(np.concatenate(([0], pred_series, [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations = ends - starts
            
            # Remove short segments
            for i, (start, end, duration) in enumerate(zip(starts, ends, durations)):
                if duration < min_duration:
                    processed[b, start:end, s] = 0
    
    return processed

def validate(model, loader, device):
    model.eval()
    total_fa = 0
    total_miss = 0
    total_speech = 0
    threshold = 0.6  # Much higher threshold to reduce false alarms
    
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            
            # Forward pass
            logits = model(feats, mask)
            probs = logits.sigmoid()
            
            # Apply higher threshold
            preds = (probs > threshold).cpu().numpy()
            
            # Apply post-processing to filter predictions
            preds = post_process_predictions(preds, min_duration=3)
            
            refs = labels.cpu().numpy()
            
            # Calculate metrics
            fa = np.sum((preds == 1) & (refs == 0))
            miss = np.sum((preds == 0) & (refs == 1))
            speech = np.sum(refs)
            
            total_fa += fa
            total_miss += miss
            total_speech += speech
            
            # Debug info
            print(f"\nValidation Stats:")
            print(f"False Alarms: {fa}")
            print(f"Misses: {miss}")
            print(f"Total Speech: {speech}")
            print(f"Predictions Active: {np.sum(preds == 1)}")
            
    der = (total_fa + total_miss) / max(total_speech, 1e-6)
    return der

def main():
    run = wandb.init(project="WhisperVox", config={
        "epochs": 1,  # Test run with single epoch
        "batch_size": 32,  # Further reduced for stability
        "lr": 1e-5,  # Reduced learning rate
        "warmup_steps": 200,  # More warmup steps
        "num_speakers": None
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define gradient accumulation steps
    global ACCUMULATION_STEPS
    ACCUMULATION_STEPS = 4  # Increased from 2
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Partially unfreeze Whisper layers in a more controlled way
    whisper_layers = list(whisper_model.model.encoder.parameters())
    total_layers = len(whisper_layers)
    # Unfreeze the last 25% of layers
    for i, param in enumerate(whisper_layers):
        param.requires_grad = (i > total_layers * 0.75)

    # Load dataset
    ds = load_dataset("diarizers-community/voxconverse")
    train_subset = ds['dev']  # Use all dev set files
    val_subset = ds['test'].select(range(5))

    # Determine speaker count
    spk_set = set()
    for item in train_subset:
        for spk in item['speakers']:
            spk_set.add(spk)
    wandb.config.update({"num_speakers": len(spk_set)}, allow_val_change=True)

    train_data = VoxSimpleDataset(train_subset, processor, run.config.num_speakers)
    val_data = VoxSimpleDataset(val_subset, processor, run.config.num_speakers)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=run.config.batch_size, 
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=False  # Reduce memory pressure
    )
    val_loader = DataLoader(val_data, batch_size=run.config.batch_size, shuffle=False)

    model = DiarizationModel(whisper_model, run.config.num_speakers).to(device)
    
    # Use a lower learning rate and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=run.config.lr,
        weight_decay=0.01,  # Add weight decay for regularization
        eps=1e-8
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=run.config.lr * 5,  # Allow for a higher peak LR
        epochs=run.config.epochs,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        pct_start=0.3,  # Longer warmup
        div_factor=25.0,  # More aggressive LR reduction
        final_div_factor=1000.0
    )
    
    scaler = torch.amp.GradScaler()

    # Training loop
    best_der = float('inf')
    for epoch in range(run.config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, run)
        der = validate(model, val_loader, device)
        
        # Log metrics
        wandb.log({
            "epoch": epoch+1, 
            "train_loss": train_loss, 
            "val_der": der
        })
        
        print(f"Epoch {epoch+1}, Loss: {train_loss:.3f}, DER: {der:.3f}")
        
        # Save best model
        if der < best_der:
            best_der = der
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_der': best_der,
            }, 'best_whisper_vox_model.pt')
            print(f"Saved best model with DER: {best_der:.3f}")

    run.finish()

if __name__ == "__main__":
    main()
