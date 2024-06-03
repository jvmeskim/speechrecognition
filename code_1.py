import librosa
import numpy as np
import os
import threading
import logging
import json
from collections import OrderedDict
import speech_recognition as sr

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the recognizer and threading lock
recognizer = sr.Recognizer()
lock = threading.Lock()

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    features = {
        'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1).tolist(),
        'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).tolist()
    }
    return features


# Function to convert speech to text
def speech_to_text(audio_file):
    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        logging.error(f"Could not understand audio file {audio_file}")
        return None
    except sr.RequestError as e:
        logging.error(f"API request failed with error: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

def process_audio_files(audio_files, results):
    local_results = {}
    for audio_file in audio_files:
        transcription = speech_to_text(audio_file)
        features = extract_audio_features(audio_file)  # Extract audio features
        if transcription:
            file_index = int(os.path.splitext(os.path.basename(audio_file))[0].split('-')[-1])
            local_results[file_index] = {
                'transcription': transcription,
                'features': features
            }
            logging.info(f"Transcribed {audio_file}: {transcription[:30]}...")
        else:
            logging.warning(f"Failed to transcribe {audio_file}")
    with lock:
        results.update(local_results)

def save_results_to_json(results, filename='transcriptions.json'):
    ordered_results = OrderedDict(sorted(results.items()))
    with open(filename, 'w') as file:
        json.dump(ordered_results, file, ensure_ascii=False, indent=4)

def main():
    base_path = '26-496-'
    audio_files = [f"{base_path}{str(i).zfill(4)}.flac" for i in range(0, 27)]
    chunks = [audio_files[i:i + 5] for i in range(0, len(audio_files), 5)]
    threads = []
    results = {}

    for chunk in chunks:
        thread = threading.Thread(target=process_audio_files, args=(chunk, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    save_results_to_json(results)
    logging.info("Finished processing all audio files.")

if __name__ == "__main__":
    main()
