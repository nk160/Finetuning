import os
import subprocess

# Create directories
os.makedirs('./data/LibriSpeech', exist_ok=True)

# URLs for train-clean-100 and dev-clean (validation)
urls = {
    'train-clean-100': 'https://www.openslr.org/resources/12/train-clean-100.tar.gz',
    'dev-clean': 'https://www.openslr.org/resources/12/dev-clean.tar.gz'
}

def download_file(url, output_dir):
    print(f"Downloading {url}...")
    # Use wget with continue flag
    cmd = f"wget -c {url} -P {output_dir}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Extract after download
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)
    print(f"Extracting {filename}...")
    cmd = f"tar -xzf {filepath} -C {output_dir}"
    subprocess.run(cmd, shell=True, check=True)

# Download and extract each file
for name, url in urls.items():
    try:
        download_file(url, './data/LibriSpeech')
    except Exception as e:
        print(f"Error downloading {name}: {str(e)}") 