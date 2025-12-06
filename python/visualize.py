import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import argparse
from scipy.signal import spectrogram

def load_wav(path):
    try:
        data, sr = sf.read(path)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data, sr
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None, None

def plot_comparison(dry_path, wet_path):
    print(f"Comparing '{dry_path}' vs '{wet_path}'...")
    
    dry, sr1 = load_wav(dry_path)
    wet, sr2 = load_wav(wet_path)
    
    if dry is None or wet is None: return

    # Ensure lengths match for plotting (trim to shorter)
    min_len = min(len(dry), len(wet))
    dry = dry[:min_len]
    wet = wet[:min_len]
    
    # Time axis
    time = np.linspace(0, min_len / sr1, min_len)

    # Setup Plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # 1. Waveform Comparison
    axs[0].set_title("Time Domain: Amplitude Decay")
    axs[0].plot(time, wet, label='Wet (Reverb)', color='dodgerblue', alpha=0.7)
    axs[0].plot(time, dry, label='Dry (Source)', color='orange', alpha=0.6, linestyle='--')
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, alpha=0.3)

    # 2. Wet Spectrogram (Frequency Content)
    axs[1].set_title("Wet Signal Spectrogram (Frequency Decay)")
    Pxx, freqs, bins, im = axs[1].specgram(wet, NFFT=1024, Fs=sr1, noverlap=512, cmap='inferno')
    axs[1].set_ylabel("Frequency (Hz)")
    
    # 3. Energy Decay (dB)
    # Calculate simple envelope
    window_size = int(sr1 * 0.01) # 10ms window
    dry_env = np.convolve(dry**2, np.ones(window_size)/window_size, mode='same')
    wet_env = np.convolve(wet**2, np.ones(window_size)/window_size, mode='same')
    
    # Avoid log(0)
    dry_db = 10 * np.log10(dry_env + 1e-12)
    wet_db = 10 * np.log10(wet_env + 1e-12)
    
    axs[2].set_title("Energy Decay Curve (dB)")
    axs[2].plot(time, wet_db, label='Wet Energy', color='dodgerblue')
    axs[2].plot(time, dry_db, label='Dry Energy', color='orange', linestyle='--')
    axs[2].set_ylabel("Power (dB)")
    axs[2].set_ylim(-60, 0) # Focus on top 60dB
    axs[2].legend(loc="upper right")
    axs[2].grid(True, alpha=0.3)
    axs[2].set_xlabel("Time (s)")

    output_img = "acoustic_analysis.png"
    plt.savefig(output_img)
    print(f"Analysis saved to: {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dry", help="Path to dry wav")
    parser.add_argument("wet", help="Path to wet wav")
    args = parser.parse_args()
    
    plot_comparison(args.dry, args.wet)