from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Setup for voice-to-text
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

def transcribe_audio(audio):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language='en').input_features
    inputs = inputs.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Setup for text-to-handwriting
def text_to_handwriting(text, font_path="DancingScript-VariableFont_wght.ttf", font_size=32):
    font = ImageFont.truetype(font_path, font_size)
    dummy_img = Image.new('RGB', (1, 1), (255, 255, 255))
    draw = ImageDraw.Draw(dummy_img)
    text_size = draw.textbbox((0, 0), text, font=font)[2:]
    img = Image.new('RGB', text_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(0, 0, 0))
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2.imwrite('handwritten_text.png', open_cv_image)
    cv2.imshow("Handwritten Text", open_cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Real-time voice-to-handwriting loop
def voice_to_handwriting(duration=5, fs=16000):
    audio_data = record_audio(duration, fs)
    transcription = transcribe_audio(audio_data)
    print("Transcription:", transcription)
    text_to_handwriting(transcription)

if __name__ == "__main__":
    # Continuously run this for a specified duration or number of iterations
    while True:
        voice_to_handwriting(duration=5)
