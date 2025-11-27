""""
AI Generated Script for Testing Audio Processing
"""
import numpy as np
import soundfile as sf

def generate_techno(duration=4.0, bpm=128, sample_rate=44100):
    print(f"Generating {duration}s of Techno at {bpm} BPM...")
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    # Beat positions (Every 1/4 note)
    seconds_per_beat = 60.0 / bpm
    
    for i in range(int(duration / seconds_per_beat)):
        start_time = i * seconds_per_beat
        # Only play kick on 1, 2, 3, 4
        start_idx = int(start_time * sample_rate)
        
        # Kick Length (0.15s)
        k_len = int(0.15 * sample_rate)
        if start_idx + k_len >= len(audio): continue
            
        # Frequency sweep (150Hz down to 50Hz)
        kt = np.linspace(0, 0.15, k_len)
        freq = np.linspace(150, 50, k_len)
        kick = np.sin(2 * np.pi * freq * kt)
        
        # Envelope (Punchy decay)
        env = np.exp(-20 * kt)
        
        audio[start_idx:start_idx+k_len] += kick * env

    for i in range(int(duration / (seconds_per_beat / 2))):
        if i % 2 == 0: continue # Skip downbeats
        
        start_time = i * (seconds_per_beat / 2)
        start_idx = int(start_time * sample_rate)
        
        h_len = int(0.05 * sample_rate) # Short burst
        if start_idx + h_len >= len(audio): continue
            
        noise = np.random.uniform(-0.5, 0.5, h_len)
        env = np.exp(-40 * np.linspace(0, 0.05, h_len))
        
        audio[start_idx:start_idx+h_len] += noise * env * 0.4

    # normalize
    audio = audio / np.max(np.abs(audio))
    return audio, sample_rate


data, sr = generate_techno()
sf.write('techno_dry.wav', data, sr)
print("Created 'techno_dry.wav'")
