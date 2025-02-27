from torch.utils.data import Dataset
import torch
import torchaudio
from pathlib import Path
from pyannote.audio import Pipeline
import os
from transformers import WhisperFeatureExtractor

class VoxConverseDataset(Dataset):
    def __init__(self, root_dir, split="dev", max_length=30, max_speakers=4, debug=False):
        self.root_dir = Path(root_dir)
        self.audio_dir = self.root_dir / "audio"
        self.split_dir = self.root_dir / split
        self.max_length = max_length
        self.max_speakers = max_speakers
        self.debug = debug
        
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
        
        # Store RTTM file paths
        self.rttm_files = {
            rttm_file.stem: rttm_file 
            for rttm_file in self.split_dir.glob("*.rttm")
            if (self.audio_dir / f"{rttm_file.stem}.wav").exists()
        }
    
    def __len__(self):
        return len(self.file_ids)
    
    def read_rttm(self, rttm_path):
        """Read RTTM file and return list of (start, end, speaker) tuples."""
        segments = []
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # RTTM format check
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    segments.append((start, start + duration, speaker))
        if not segments:
            raise ValueError(f"No valid segments found in {rttm_path}")
        return sorted(segments)
    
    def find_best_window(self, segments, window_size):
        """Find the window with the most balanced speaker distribution."""
        if self.debug:
            print("\nSearching for best window...")
        
        best_window = None
        best_score = -float('inf')
        best_start = 0
        best_durations = None
        
        # Special handling for single-speaker files
        unique_speakers = len(set(spk for _, _, spk in segments))
        if unique_speakers == 1:
            if self.debug:
                print("Single speaker file - using first window")
            # Take first window_size seconds of speech
            window_segments = [
                (s, e, spk) for s, e, spk in segments 
                if s < window_size
            ]
            return window_segments if window_segments else [segments[0]]
        
        # Multi-speaker window search
        for start in range(0, int(segments[-1][1] - window_size), 5):
            end = start + window_size
            window_segments = [
                (s, e, spk) for s, e, spk in segments 
                if (s >= start and s < end) or (e > start and e <= end) or (s <= start and e >= end)
            ]
            
            # Calculate speaker durations in this window
            speaker_durations = {}
            for s, e, spk in window_segments:
                seg_start = max(s, start)
                seg_end = min(e, start + window_size)
                duration = seg_end - seg_start
                speaker_durations[spk] = speaker_durations.get(spk, 0) + duration
            
            num_speakers = len(speaker_durations)
            if num_speakers < 2:
                continue
            
            # Score based on number of speakers and balance
            durations = list(speaker_durations.values())
            min_duration = min(durations)
            max_duration = max(durations)
            balance_ratio = min_duration / max_duration if max_duration > 0 else 0
            
            # Require minimum speaking time and good balance
            if min_duration < 5.0:  # At least 5 seconds per speaker
                continue
                
            # New scoring formula heavily prioritizing balance
            score = balance_ratio * 30 + num_speakers * 5  # More weight on balance
            
            if score > best_score:
                best_score = score
                best_window = window_segments
                best_start = start
                best_durations = speaker_durations.copy()
                
                if self.debug:
                    print(f"Found better window at {start}s with {num_speakers} speakers")
                    print(f"Speaker durations: {speaker_durations}")
                    print(f"Balance ratio: {balance_ratio:.2f}, Score: {score:.2f}")
        
        if self.debug:
            print(f"\nBest window starts at {best_start}s")
            if best_window:
                print(f"Contains {len(set(spk for _,_,spk in best_window))} speakers")
                print(f"Speaker durations: {best_durations}")
        
        return best_window or segments[:1]  # Return first segment if no multi-speaker window found
    
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        if self.debug:
            print(f"\nProcessing file: {file_id}")
        
        # Load audio and RTTM
        audio_path = self.audio_dir / f"{file_id}.wav"
        rttm_path = self.rttm_files[file_id]
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not rttm_path.exists():
            raise FileNotFoundError(f"RTTM file not found: {rttm_path}")
            
        waveform, sample_rate = torchaudio.load(audio_path)
        segments = self.read_rttm(rttm_path)
        
        # Find best window instead of just cutting at max_length
        window_segments = self.find_best_window(segments, self.max_length)
        
        if not window_segments:
            raise ValueError(f"No valid window found in {file_id}")
        
        # Get unique speakers in order of appearance
        speaker_timeline = {}
        for start, _, speaker in window_segments:
            if speaker not in speaker_timeline:
                speaker_timeline[speaker] = start
        
        speakers = sorted(speaker_timeline.keys(), key=lambda x: speaker_timeline[x])
        
        if self.debug:
            print(f"Found speakers in order of appearance: {speakers}")
            print(f"Speaker timeline: {speaker_timeline}")
        
        # Map ALL speakers to indices (up to max_speakers)
        num_speakers = min(len(speakers), self.max_speakers)
        speaker_map = {speaker: i for i, speaker in enumerate(speakers[:num_speakers])}
        
        if self.debug:
            print(f"Using {num_speakers} speakers with mapping: {speaker_map}")
        
        # Process audio features
        features = self.feature_extractor(
            waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.squeeze(0)  # [80, 3000]
        
        num_frames = features.shape[1]  # 3000
        encoder_frames = num_frames // 2  # 1500
        
        # Create labels from RTTM segments
        labels = torch.zeros(encoder_frames, self.max_speakers)  # [1500, 4]
        
        # Track which speakers are actually used
        used_speakers = set()
        
        for start, end, speaker in window_segments:
            if speaker in speaker_map:
                start_frame = min(int(start * 25), encoder_frames-1)  # 25 fps
                end_frame = min(int(end * 25), encoder_frames)
                speaker_idx = speaker_map[speaker]
                labels[start_frame:end_frame, speaker_idx] = 1
                used_speakers.add(speaker)
        
        if self.debug:
            print(f"\nActually used speakers: {used_speakers}")
        
        # Upsample labels to match features
        labels = torch.nn.functional.interpolate(
            labels.transpose(0, 1).unsqueeze(0),  # [1, 4, 1500]
            size=num_frames,  # 3000
            mode='nearest'
        ).squeeze(0).transpose(0, 1)  # Back to [3000, 4]
        
        if self.debug:
            print(f"Features shape: {features.shape}")  # Should be [80, 3000]
            print(f"Labels shape: {labels.shape}")      # Should be [3000, 4]
        
        # Add shape assertions
        assert features.shape[0] == 80, f"Expected 80 mel bins, got {features.shape[0]}"
        assert labels.shape[1] == self.max_speakers, f"Expected {self.max_speakers} speakers, got {labels.shape[1]}"
        assert features.shape[1] == labels.shape[0], f"Time dimension mismatch: features={features.shape[1]}, labels={labels.shape[0]}"
        
        # After creating labels
        if self.debug:
            print(f"\nLabel statistics:")
            print(f"Total frames: {labels.shape[0]}")
            print(f"Positive labels: {labels.sum()}")
            print(f"Label distribution per speaker: {labels.sum(dim=0)}")
            print(f"Sample of labels:\n{labels[0:10, :]}\n")
        
        # Debug diarization output
        if self.debug:
            print("\nDiarization segments:")
            total_duration = 0
            speaker_durations = {}
            for start, end, speaker in window_segments:  # Unpack tuple values
                duration = end - start  # Use tuple values directly
                total_duration += duration
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
                print(f"Time: {start:.1f}s -> {end:.1f}s, Speaker: {speaker}")
            
            print("\nSpeaker statistics:")
            for speaker, duration in speaker_durations.items():
                percentage = (duration / total_duration) * 100
                print(f"{speaker}: {duration:.1f}s ({percentage:.1f}%)")
        
        return {
            "input_features": features,      # [80, 3000]
            "diarization_labels": labels,    # [3000, 4]
            "speaker_map": speaker_map
        } 