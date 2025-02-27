import torch
from transformers import WhisperPreTrainedModel, WhisperConfig, WhisperModel
from torch import nn

class WhisperForDiarization(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.whisper = WhisperModel(config)
        
        # Diarization head: predicts speaker embeddings for each time step
        self.diarization_head = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_speakers)  # Output per-speaker probabilities
        )
        
    def forward(
        self,
        input_features,
        attention_mask=None,
        diarization_labels=None,
    ):
        # Get Whisper encoder outputs
        outputs = self.whisper.encoder(
            input_features,
            attention_mask=attention_mask,
        )
        
        # Apply diarization head
        diarization_logits = self.diarization_head(outputs.last_hidden_state)
        
        loss = None
        if diarization_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(diarization_logits, diarization_labels)
            
        return {
            "loss": loss,
            "diarization_logits": diarization_logits,
            "hidden_states": outputs.hidden_states,
        } 