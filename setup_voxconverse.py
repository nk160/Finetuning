import os
from pathlib import Path
import subprocess
import shutil

def setup_voxconverse():
    """Set up VoxConverse dataset structure"""
    # Create base directory
    base_dir = Path("voxconverse")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (base_dir / "audio").mkdir(exist_ok=True)
    (base_dir / "dev").mkdir(exist_ok=True)
    (base_dir / "test").mkdir(exist_ok=True)
    
    # Clone VoxConverse repository
    if not (base_dir / "repo").exists():
        subprocess.run([
            "git", "clone",
            "https://github.com/joonson/voxconverse.git",
            str(base_dir / "repo")
        ])
    
    # Copy RTTM files to appropriate directories
    repo_dir = base_dir / "repo"
    
    # Copy dev set RTTMs
    for rttm in (repo_dir / "dev").glob("*.rttm"):
        shutil.copy2(rttm, base_dir / "dev" / rttm.name)
    
    # Copy test set RTTMs
    for rttm in (repo_dir / "test").glob("*.rttm"):
        shutil.copy2(rttm, base_dir / "test" / rttm.name)
    
    print("VoxConverse directory structure created")
    print("Now download audio files from the official source and place them in voxconverse/audio/")

if __name__ == "__main__":
    setup_voxconverse() 