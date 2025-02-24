import torch
import whisper
import wandb

# Initialize wandb
wandb.init(
    project="whisper-fine-tuning",
    name="whisper-5epochs",  # Different name to distinguish from 10-epoch version
    config={
        "epochs": 5,  # Changed to 5
        "learning_rate": 1e-5,
        "model_type": "tiny.en",
        "ground_truth": "Hello, my name is Izaak."
    }
)

# Load the model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load the audio file and test the model
audio_path = "hello.wav"
result = model.transcribe(audio_path)
initial_transcription = result["text"]  # Store initial transcription
print("Transcription:", initial_transcription)

# Process the audio file
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio)
mel = mel.to(device)
mel = mel.unsqueeze(0)

# Tokenize ground truth text
ground_truth_text = "Hello, my name is Izaak."
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)
target_ids = tokenizer.encode(ground_truth_text)
sot_token = torch.tensor([[tokenizer.sot]], dtype=torch.long, device=device)
target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
input_tks = torch.cat([sot_token, target_tensor], dim=-1)

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Modified training loop with wandb logging
model.train()
for step in range(5):  # 5 epochs
    predictions = model(tokens=input_tks, mel=mel)
    remove_sot = input_tks[:, 1:]
    predictions = predictions[:, :-1, :]
    loss = criterion(predictions.transpose(1, 2), remove_sot)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log metrics to wandb
    wandb.log({
        "epoch": step + 1,
        "loss": loss.item(),
    })
    print(f"Step {step+1}/5, Loss: {loss.item():.4f}")

# Test and log final results
model.eval()
torch.set_grad_enabled(False)
result = model.transcribe(audio_path)
wandb.log({
    "final_transcription": result["text"],
    "initial_transcription": initial_transcription
})

# Save model weights as W&B Artifact
model_artifact = wandb.Artifact(
    name=f"whisper_model_5epoch",
    type="model",
    description="Whisper model fine-tuned for 5 epochs"
)

# Save the model weights
model_path = "whisper_5epoch.pt"
torch.save({
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, model_path)

# Add the model file to the artifact
model_artifact.add_file(model_path)

# Log the artifact to W&B
wandb.log_artifact(model_artifact)

# Close wandb run
wandb.finish()