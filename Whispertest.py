import torch
import whisper

# Load the model
model = whisper.load_model("tiny.en")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load the audio file and test the model
audio_path = "hello.wav"
result = model.transcribe(audio_path)
print("Transcription:", result["text"])

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

# Train the model
model.train()
for step in range(5):
    # Forward pass
    predictions = model(tokens=input_tks, mel=mel) # teacher forcing!
    remove_sot = input_tks[:, 1:] # remove SOT token to align targets with predictions
    predictions = predictions[:, :-1, :] # remove the last prediction again for alignment
    loss = criterion(predictions.transpose(1, 2), remove_sot)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step+1}/5, Loss: {loss.item():.4f}")

# Test the model
model.eval()
torch.set_grad_enabled(False)
result = model.transcribe(audio_path)
print("Transcription:", result["text"])