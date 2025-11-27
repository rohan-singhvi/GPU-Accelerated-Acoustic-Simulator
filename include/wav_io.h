#ifndef WAV_IO_H
#define WAV_IO_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstring>

struct WavData {
    std::vector<float> samples;
    int sample_rate;
    bool success;
};

inline void write_wav(const std::string& filename, const std::vector<float>& buffer, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    
    float max_val = 0.0f;
    for (float f : buffer) max_val = std::max(max_val, std::abs(f));
    
    // normalize to prevent clipping, but only if we exceed 1.0 or if it's very quiet
    float scale = 1.0f;
    if (max_val > 1.0f) scale = 0.99f / max_val;
    
    std::vector<int16_t> int_data(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        // clamp to -1.0 to 1.0 before converting
        float val = buffer[i] * scale;
        val = std::max(-1.0f, std::min(1.0f, val));
        int_data[i] = static_cast<int16_t>(val * 32767.0f);
    }

    int num_channels = 1;
    int bits_per_sample = 16;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int block_align = num_channels * bits_per_sample / 8;
    int sub_chunk2_size = int_data.size() * num_channels * bits_per_sample / 8;
    int chunk_size = 36 + sub_chunk2_size;

    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&chunk_size), 4);
    file.write("WAVE", 4);
    file.write("fmt ", 4);
    
    int sub_chunk1_size = 16;
    short audio_format = 1; 
    short num_channels_short = (short)num_channels;
    
    file.write(reinterpret_cast<const char*>(&sub_chunk1_size), 4);
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels_short), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&sub_chunk2_size), 4);
    file.write(reinterpret_cast<const char*>(int_data.data()), int_data.size() * sizeof(int16_t));
    
    std::cout << "Saved " << filename << std::endl;
}


inline WavData read_wav(const std::string& filename) {
    WavData result;
    result.success = false;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return result;
    }

    char buffer[4];
    file.read(buffer, 4); // riff
    if (strncmp(buffer, "RIFF", 4) != 0) return result;
    
    file.seekg(4, std::ios::cur); // skip
    file.read(buffer, 4); // wave
    if (strncmp(buffer, "WAVE", 4) != 0) return result;

    // search for "fmt " and "data" chunks
    int sample_rate = 0;
    short num_channels = 0;
    short bits_per_sample = 0;
    
    while (file.read(buffer, 4)) {
        int chunk_size;
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        
        if (strncmp(buffer, "fmt ", 4) == 0) {
            short audio_format;
            file.read(reinterpret_cast<char*>(&audio_format), 2);
            file.read(reinterpret_cast<char*>(&num_channels), 2);
            file.read(reinterpret_cast<char*>(&sample_rate), 4);
            file.seekg(6, std::ios::cur); // skip ByteRate, BlockAlign
            file.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            
            // skip any extra fmt bytes
            if (chunk_size > 16) file.seekg(chunk_size - 16, std::ios::cur);
        }
        else if (strncmp(buffer, "data", 4) == 0) {
            // read audio data
            if (bits_per_sample != 16) {
                std::cerr << "Error: Only 16-bit PCM WAV supported." << std::endl;
                return result;
            }
            
            int num_samples = chunk_size / (num_channels * 2);
            result.samples.resize(num_samples);
            
            // read into temp buffer then convert to float
            std::vector<int16_t> temp_data(num_samples * num_channels);
            file.read(reinterpret_cast<char*>(temp_data.data()), chunk_size);
            
            // convert to mono (mixdown if stereo)
            for (int i = 0; i < num_samples; ++i) {
                float sum = 0.0f;
                for (int c = 0; c < num_channels; ++c) {
                    sum += temp_data[i * num_channels + c];
                }
                result.samples[i] = (sum / num_channels) / 32768.0f;
            }
            
            result.sample_rate = sample_rate;
            result.success = true;
            break; 
        }
        else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }
    
    return result;
}

#endif