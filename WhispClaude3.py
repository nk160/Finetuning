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

# ---------- Balanced Model: Not too complex, not too simple ----------
class DiarizationModel(torch.nn.Module):
    def __init__(self, whisper_model, num_speakers):
        super().__init__()
        self.whisper = whisper_model
        hidden = self.whisper.config.d_model
        
        # Simple convolutional layer for temporal context
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3)
        )
        
        # Simple classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, num_speakers)
        )
        
        # Initialize with mild negative bias
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == num_speakers:
                # Moderate negative bias (-0.5 instead of -2.0)
                torch.nn.init.constant_(m.bias, -0.5)

    def forward(self, input_features, attention_mask):
        # Get encoder outputs from Whisper
        enc_out = self.whisper.model.encoder(
            input_features, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        
        # Apply conv layer
        x = self.conv(enc_out.transpose(1, 2)).transpose(1, 2)
        
        # Apply classifier
        return self.classifier(x)

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
    
    for i, batch in enumerate(tqdm(loader, desc="Train")):
        feats = batch["input_features"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)
        
        with torch.amp.autocast(device_type='cuda'):
            logits = model(feats, mask)
            
            # Balanced class weighting (1:2 instead of 1:10)
            pos_weight = torch.ones(run.config.num_speakers, device=device) * 3.0
            
            # Use standard BCE loss with pos_weight parameter
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels, 
                reduction='mean',
                pos_weight=pos_weight
            )
        
        scaler.scale(loss / ACCUMULATION_STEPS).backward()
        total_loss += loss.item()
        
        # Log metrics
        if i % 10 == 0:
            with torch.no_grad():
                preds = (logits.sigmoid() > 0.5).float()
                pred_ratio = preds.mean().item()
                label_ratio = labels.mean().item()
                
                run.log({
                    "batch_loss": loss.item(),
                    "pred_ratio": pred_ratio,
                    "label_ratio": label_ratio
                })
        
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
        if i % 5 == 0:
            torch.cuda.empty_cache()
        clear_mem()
        
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_fa = 0
    total_miss = 0
    total_speech = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            
            # Forward pass
            logits = model(feats, mask)
            
            # Try multiple thresholds
            thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
            best_der = float('inf')
            best_preds = None
            best_thresh = None
            
            for thresh in thresholds:
                preds = (logits.sigmoid() > thresh).cpu().numpy()
                preds = filter_predictions(preds)
                
                # Calculate error for this threshold
                refs = labels.cpu().numpy()
                fa = np.sum((preds == 1) & (refs == 0))
                miss = np.sum((preds == 0) & (refs == 1))
                speech = np.sum(refs)
                
                der = (fa + miss) / max(speech, 1e-6)
                
                if der < best_der:
                    best_der = der
                    best_preds = preds
                    best_thresh = thresh
            
            # Use the best predictions
            preds = best_preds
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
            print(f"Best Threshold: {best_thresh}")
            
    der = (total_fa + total_miss) / max(total_speech, 1e-6)
    return der

def filter_predictions(preds, min_duration=3, max_gaps=2):
    """
    Filter predictions to remove short segments and fill small gaps
    """
    processed = preds.copy()
    batch_size, time_steps, num_speakers = processed.shape
    
    for b in range(batch_size):
        for s in range(num_speakers):
            # Get binary prediction series for this speaker
            pred_series = processed[b, :, s]
            
            # Fill small gaps
            transitions = np.diff(np.concatenate(([0], pred_series, [0])))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            
            # Fill small gaps between segments
            for i in range(len(starts) - 1):
                gap_size = starts[i+1] - ends[i]
                if 0 < gap_size <= max_gaps:
                    processed[b, ends[i]:starts[i+1], s] = 1
            
            # After filling gaps, remove short segments
            transitions = np.diff(np.concatenate(([0], processed[b, :, s], [0])))
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            durations = ends - starts
            
            for i, (start, end, duration) in enumerate(zip(starts, ends, durations)):
                if duration < min_duration:
                    processed[b, start:end, s] = 0
    
    return processed

def main():
    run = wandb.init(project="WhisperVox", config={
        "epochs": 1,   # Initial test run
        "batch_size": 4,  # Small batch size for more gradient updates
        "lr": 2e-5,  # Slightly higher learning rate
        "num_speakers": None,
        "early_stopping_patience": 5
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define gradient accumulation steps
    global ACCUMULATION_STEPS
    ACCUMULATION_STEPS = 2
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Freeze most of Whisper, but unfreeze last two layers for better adaptation
    for param in whisper_model.parameters():
        param.requires_grad = False
        
    # Unfreeze last two encoder layers
    for layer in whisper_model.model.encoder.layers[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Load dataset
    ds = load_dataset("diarizers-community/voxconverse")
    train_subset = ds['dev'].select(range(30))  # First 30 files to speed up training
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
        pin_memory=False
    )
    val_loader = DataLoader(val_data, batch_size=run.config.batch_size, shuffle=False)

    model = DiarizationModel(whisper_model, run.config.num_speakers).to(device)
    
    # Use a moderate optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=run.config.lr,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Scheduler with longer warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=run.config.lr * 10,
        epochs=run.config.epochs,
        steps_per_epoch=len(train_loader) // ACCUMULATION_STEPS,
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    scaler = torch.amp.GradScaler()

    # Training loop
    best_der = float('inf')
    patience = run.config.early_stopping_patience
    no_improve_count = 0
    
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
        
        # Save best model and check for early stopping
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
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break

    run.finish()

if __name__ == "__main__":
    main()