""""
AI Generated Script for Testing Audio Processing
"""
import numpy as np
import soundfile as sf
import argparse

def generate_techno_dry(duration=4.0, bpm=120, sample_rate=44100):
    print(f"Generating {duration}s of Sparse Techno at {bpm} BPM...")
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t)
    
    seconds_per_beat = 60.0 / bpm
    
    #Sine sweep - Occurs on Beat 1
    # We leave Beat 2 empty to hear the room
    for i in range(0, int(duration / seconds_per_beat), 2): 
        start_time = i * seconds_per_beat
        start_idx = int(start_time * sample_rate)
        
        k_len = int(0.15 * sample_rate)
        if start_idx + k_len >= len(audio): continue
            
        kt = np.linspace(0, 0.15, k_len)
        freq = np.linspace(150, 40, k_len) # Sweep down
        kick = np.sin(2 * np.pi * freq * kt)
        env = np.exp(-15 * kt)
        audio[start_idx:start_idx+k_len] += kick * env * 0.8

    # clap - Occurs on Beat 3 (Backbeat)
    # We leave Beat 4 empty
    for i in range(0, int(duration / seconds_per_beat), 2):
        # Offset by 1 beat (the "and" or the "3")
        start_time = (i + 1) * seconds_per_beat 
        start_idx = int(start_time * sample_rate)
        
        c_len = int(0.08 * sample_rate)
        if start_idx + c_len >= len(audio): continue
            
        noise = np.random.uniform(-0.5, 0.5, c_len)
        # Jagged envelope for clap texture
        env = np.exp(-30 * np.linspace(0, 0.08, c_len))
        audio[start_idx:start_idx+c_len] += noise * env * 0.5

    # Normalize to -1.0 to 1.0
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        
    return audio, sample_rate

if __name__ == "__main__":
    data, sr = generate_techno_dry()
    filename = "techno_dry.wav"
    sf.write(filename, data, sr)
    print(f"Success! Saved '{filename}' (Use this as --input)")