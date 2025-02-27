import torch
from pathlib import Path
import torchaudio
from pyannote.audio import Pipeline
import os
from rich.console import Console
from rich.table import Table

def test_voxconverse_pipeline():
    console = Console()
    console.print("\n[bold blue]Testing VoxConverse Diarization Pipeline[/bold blue]")
    
    # Initialize diarization pipeline
    console.print("\n[yellow]Initializing pyannote diarization...[/yellow]")
    hf_token = os.getenv("HF_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Test files
    test_files = ["abjxc", "afjiv"]
    
    for file_id in test_files:
        console.print(f"\n[green]Processing file: {file_id}[/green]")
        
        # Load files
        audio_path = Path(f"voxconverse/audio/{file_id}.wav")
        rttm_path = Path(f"voxconverse/dev/{file_id}.rttm")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sample_rate
            console.print(f"Audio duration: {duration:.1f} seconds")
            
            # Run diarization
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
            
            # Create results table
            table = Table(title=f"Diarization Results for {file_id}")
            table.add_column("Time", style="cyan")
            table.add_column("Speaker", style="magenta")
            table.add_column("Duration", style="green")
            
            # Get unique speakers and their total speaking time
            speaker_times = {}
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                duration = segment.end - segment.start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
                
                # Add first 5 segments to table
                if len(table.rows) < 5:
                    table.add_row(
                        f"{segment.start:.1f}s → {segment.end:.1f}s",
                        speaker,
                        f"{duration:.1f}s"
                    )
            
            # Print results
            console.print(f"\nFound {len(speaker_times)} speakers:")
            for speaker, total_time in speaker_times.items():
                console.print(f"- {speaker}: {total_time:.1f} seconds total")
            
            console.print("\nFirst 5 speaker segments:")
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error processing file {file_id}: {str(e)}[/red]")
    
    console.print("\n[bold green]Test completed![/bold green]")
    return True

if __name__ == "__main__":
    test_voxconverse_pipeline() 