from torch.utils.data import Dataset
import torch
import torchaudio
from pathlib import Path
from pyannote.audio import Pipeline
import os
from transformers import WhisperFeatureExtractor

class VoxConverseDataset(Dataset):
    def __init__(self, root_dir, split="dev", max_length=30):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / "audio"
        self.split_dir = self.root_dir / split
        self.max_length = max_length
        
        # Initialize diarization pipeline
        self.diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Get all matching audio-RTTM pairs
        self.file_ids = []
        for rttm_file in self.split_dir.glob("*.rttm"):
            file_id = rttm_file.stem
            if (self.audio_dir / f"{file_id}.wav").exists():
                self.file_ids.append(file_id)
        
        # Add feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # Load audio
        audio_path = self.audio_dir / f"{file_id}.wav"
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Get diarization segments
        diarization = self.diarization({"waveform": waveform, "sample_rate": sample_rate})
        
        # Convert diarization to frame-level labels
        # Each frame will have a binary vector indicating which speakers are active
        num_frames = int(waveform.shape[1] / (sample_rate * 0.02))  # 20ms frames
        speakers = list(set(speaker for _, _, speaker in diarization.itertracks(yield_label=True)))
        speaker_map = {speaker: i for i, speaker in enumerate(speakers)}
        
        # Create frame-level speaker labels
        labels = torch.zeros(num_frames, len(speakers))
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_frame = int(segment.start / 0.02)
            end_frame = int(segment.end / 0.02)
            speaker_idx = speaker_map[speaker]
            labels[start_frame:end_frame, speaker_idx] = 1
        
        # Process audio into features
        features = self.feature_extractor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        return {
            "input_features": features.input_features.squeeze(0),
            "diarization_labels": labels,
            "speaker_map": speaker_map
        } 