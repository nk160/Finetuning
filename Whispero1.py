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

# ---------- Model Without LSTM ----------
class DiarizationModel(torch.nn.Module):
    def __init__(self, whisper_model, num_speakers):
        super().__init__()
        self.whisper = whisper_model
        hidden = self.whisper.config.d_model
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.GroupNorm(8, hidden),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden, hidden, 3, padding=1),
            torch.nn.GroupNorm(8, hidden),
            torch.nn.GELU()
        )
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden, num_heads=4, batch_first=True
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, num_speakers)
        )
        self.temporal = torch.nn.LSTM(
            hidden, hidden, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )

    def forward(self, input_features, attention_mask):
        enc_out = self.whisper.model.encoder(
            input_features, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        # Conv block
        x = self.conv_block(enc_out.transpose(1, 2)).transpose(1, 2)
        # Self-attention
        key_pad = (~attention_mask[:, ::2].bool())  # half frames
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_pad)
        x = x + attn_out
        # Final classification
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
    for i, batch in enumerate(tqdm(loader, desc="Train")):
        feats = batch["input_features"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].float().to(device)
        with torch.amp.autocast(device_type='cuda'):
            logits = model(feats, mask)
            probs = logits.sigmoid()
            pt = torch.where(labels == 1, probs, 1 - probs)
            focal_weight = (1 - pt) ** 2
            loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, 
                labels, 
                reduction='none',
                pos_weight=torch.ones(run.config.num_speakers, device=device) * 5.0
            )
            loss = (loss_bce * focal_weight).mean()
        scaler.scale(loss / ACCUMULATION_STEPS).backward()
        total_loss += loss.item()
        run.log({"batch_loss": loss.item()})
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        clear_mem()
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    total_fa = 0
    total_miss = 0
    total_speech = 0
    with torch.no_grad():
        for batch in loader:
            feats = batch["input_features"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].float().to(device)
            logits = model(feats, mask).sigmoid()
            preds = (logits > 0.5).cpu().numpy()
            refs = labels.cpu().numpy()
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
        "epochs": 15,
        "batch_size": 128,  # Double batch size
        "lr": 3e-5,
        "num_speakers": None
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Freeze entire Whisper encoder except final layers if desired
    for param in whisper_model.model.encoder.parameters():
        param.requires_grad = False

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
    train_loader = DataLoader(train_data, batch_size=run.config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=run.config.batch_size, shuffle=False)

    model = DiarizationModel(whisper_model, run.config.num_speakers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run.config.lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=run.config.lr,
        epochs=run.config.epochs,
        steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler()

    # And add gradient accumulation
    ACCUMULATION_STEPS = 2  # Effective batch size = 128 * 2 = 256

    for epoch in range(run.config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, run)
        der = validate(model, val_loader, device)
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_der": der})
        print(f"Epoch {epoch+1}, Loss: {train_loss:.3f}, DER: {der:.3f}")

    run.finish()

if __name__ == "__main__":
    main()

