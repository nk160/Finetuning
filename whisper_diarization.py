import torch
from transformers import WhisperPreTrainedModel, WhisperConfig, WhisperModel
from torch import nn

class WhisperDiarizationConfig(WhisperConfig):
    def __init__(self, num_speakers=2, **kwargs):
        super().__init__(**kwargs)
        self.num_speakers = num_speakers

class WhisperForDiarization(WhisperPreTrainedModel):
    def __init__(self, config: WhisperDiarizationConfig):
        super().__init__(config)
        self.whisper = WhisperModel(config)
        
        # Simpler architecture with careful initialization
        hidden_size = config.hidden_size
        
        self.diarization_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, config.num_speakers),
        )
        
        # Proper initialization for different parameter types
        for name, param in self.diarization_head.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Xavier init for weight matrices
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Zero init for biases
                nn.init.zeros_(param)
            elif 'weight' in name:
                # Normal init for 1D weights (like LayerNorm)
                nn.init.normal_(param, mean=1.0, std=0.02)
    
    def forward(self, input_features, attention_mask=None, diarization_labels=None):
        # Get encoder outputs
        outputs = self.whisper.encoder(input_features, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply diarization head
        diarization_logits = self.diarization_head(hidden_states)
        
        loss = None
        if diarization_labels is not None:
            # Focal Loss with label smoothing
            loss_fct = nn.BCEWithLogitsLoss(
                reduction='none',
                pos_weight=torch.tensor([2.0]).to(diarization_labels.device)  # Weight positive samples more
            )
            
            # Downsample labels
            diarization_labels = torch.nn.functional.interpolate(
                diarization_labels.transpose(1, 2),
                size=diarization_logits.shape[1],
                mode='nearest'
            ).transpose(1, 2)
            
            # Apply label smoothing
            diarization_labels = diarization_labels * 0.9 + 0.05
            
            # Calculate loss with focal term
            bce_loss = loss_fct(diarization_logits, diarization_labels)
            probs = torch.sigmoid(diarization_logits)
            focal_term = (1 - probs) ** 2
            loss = (focal_term * bce_loss).mean()
        
        return {"loss": loss, "diarization_logits": diarization_logits} 