import os
import asyncio
import soundfile as sf
from gtts import gTTS
import edge_tts
import librosa

REAL_DIR = "data/raw/real_samples"
FAKE_DIR = "data/raw/fake_indian_tts"
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

sentences = [
    "Hello, how are you doing today?",
    "I am calling to inform you about your car's extended warranty.",
    "Please provide your credit card details for verification.",
    "The weather is quite pleasant in Bangalore this time of year.",
    "Can you help me fix the bugs in my Python code?",
    "The stock market saw a huge dip today.",
    "Artificial intelligence is transforming the software industry."
] * 15 # 105 sentences

async def generate_real():
    print("\n[1/2] Generating highly realistic Indian English samples (Proxy for Real)...")
    count = 0
    for i, text in enumerate(sentences[:100]):
        communicate = edge_tts.Communicate(text, "en-IN-PrabhatNeural")
        temp_mp3 = os.path.join(REAL_DIR, f"temp_indian_real_{i}.mp3")
        final_wav = os.path.join(REAL_DIR, f"indian_real_{i}.wav")
        await communicate.save(temp_mp3)
        y, sr = librosa.load(temp_mp3, sr=16000)
        sf.write(final_wav, y, sr)
        os.remove(temp_mp3)
        count += 1
    print(f"[OK] Successfully generated {count} realistic Indian English samples.")

def generate_fake():
    print("\n[2/2] Generating robotic synthetic Indian English samples (gTTS)...")
    count = 0
    for i, text in enumerate(sentences[:100]):
        tts = gTTS(text, lang='en', tld='co.in')
        temp_mp3 = os.path.join(FAKE_DIR, f"temp_indian_fake_{i}.mp3")
        final_wav = os.path.join(FAKE_DIR, f"indian_fake_{i}.wav")
        tts.save(temp_mp3)
        y, sr = librosa.load(temp_mp3, sr=16000)
        sf.write(final_wav, y, sr)
        os.remove(temp_mp3)
        count += 1
    print(f"[OK] Successfully generated {count} synthetic Indian English deepfakes.")

async def main():
    print("Starting Dataset Augmentation for Indian Accents...")
    await generate_real()
    generate_fake()
    print("\nDone! Dataset successfully augmented.")

if __name__ == "__main__":
    asyncio.run(main())
