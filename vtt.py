import torch
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    torch_dtype=torch_dtype,
    device=device,
)

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished
    return audio.squeeze()

def save_audio(audio, filename="output.wav", fs=16000):
    write(filename, fs, audio)

def transcribe_audio(audio):
    # The audio should be in the correct format expected by the pipeline
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language='en').input_features
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Record audio from the microphone
audio_data = record_audio(duration=5)

# Save the recorded audio if needed (optional)
save_audio(audio_data, "output.wav")

# Transcribe the recorded audio
transcription = transcribe_audio(audio_data)
print("Transcription:", transcription)