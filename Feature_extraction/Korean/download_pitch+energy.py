import parselmouth
from parselmouth.praat import call
import numpy as np
from scipy import interpolate
import pandas as pd
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import librosa  # 이하 추가함 
import librosa.display

drive_csv_path = "/root/literature.csv"
full_data = pd.read_csv(drive_csv_path)
project_id = "research-9th-ser"
bucket_name = "research_9th_ser_data"

def extract_pitch_values(audio_file):
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch_ac(time_step=0.01, pitch_floor=50, pitch_ceiling=500)
    pitch_values = pitch.selected_array['frequency']
    time_stamps = pitch.xs()
    valid_indices = pitch_values > 0
    pitch_values = pitch_values[valid_indices]
    time_stamps = time_stamps[valid_indices]
    return pitch_values, time_stamps

def stylize_pitch_tier_from_sound(audio_file, frequency_resolution=2.0, pitch_floor=50, pitch_ceiling=500):
    try:
        sound = parselmouth.Sound(audio_file)
        manipulation = call(sound, "To Manipulation", 0.01, pitch_floor, pitch_ceiling)
        pitch_tier = call(manipulation, "Extract pitch tier")
        call(pitch_tier, "Stylize...", frequency_resolution, "semitones")
        close_copy_pitch = call(pitch_tier, "To Pitch...", 0.01, pitch_floor, pitch_ceiling)
        stylized_pitch = close_copy_pitch.selected_array['frequency']
        stylized_time = close_copy_pitch.xs()
        if len(stylized_time) == 0 or len(stylized_pitch) == 0:
            raise ValueError("Stylized time or pitch is empty.")
        return np.array(stylized_time), np.array(stylized_pitch)
    except Exception as e:
        return np.array([]), np.array([])

def get_pitch_over_threshold(audio_file, threshold=10.0):
    pitch_values, time_stamps = extract_pitch_values(audio_file)
    if len(pitch_values) == 0:
        return []
    applied_pitch = [pitch_values[0]]
    applied_time = [time_stamps[0]]
    for i in range(1, len(pitch_values)):
        if abs(pitch_values[i] - applied_pitch[-1]) >= threshold:
            applied_pitch.append(pitch_values[i])
            applied_time.append(time_stamps[i])
    applied_pitch, applied_time = np.array(applied_pitch), np.array(applied_time)
    contour_data = []
    for i in range(1, len(applied_pitch)):
        contour_data.append({'time': applied_time[i], 'pitch': applied_pitch[i]})
    return contour_data

def extract_pitch_features_with_stylization_and_distance(audio_file, threshold=10.0):
    pitch_values, time_stamps = extract_pitch_values(audio_file)
    if len(pitch_values) == 0:
        return []
    stylized_pitch = [pitch_values[0]]
    stylized_time = [time_stamps[0]]
    for i in range(1, len(pitch_values)):
        if abs(pitch_values[i] - stylized_pitch[-1]) >= threshold:
            stylized_pitch.append(pitch_values[i])
            stylized_time.append(time_stamps[i])
    stylized_pitch, stylized_time = np.array(stylized_pitch), np.array(stylized_time)
    movement_data = calculate_movement_distance_and_slope(stylized_pitch, stylized_time)
    return movement_data

def calculate_movement_distance_and_slope(pitch_values, time_stamps):
    min_pitch = np.min(pitch_values)
    max_pitch = np.max(pitch_values)
    normalized_pitch = (pitch_values - min_pitch) / (max_pitch - min_pitch) * 100
    min_time = np.min(time_stamps)
    max_time = np.max(time_stamps)
    normalized_time = (time_stamps - min_time) / (max_time - min_time) * 100
    movement_data = []
    for i in range(1, len(normalized_pitch)):
        delta_pitch = normalized_pitch[i] - normalized_pitch[i - 1]
        delta_time = normalized_time[i] - normalized_time[i - 1]
        slope = delta_pitch / delta_time if delta_time != 0 else 0
        movement_data.append({
            'end_time': normalized_time[i],
            'slope': slope
        })
    return movement_data

def calculate_slopes(stylized_time, stylized_pitch):
    slopes = []
    for i in range(1, len(stylized_time)):
        delta_pitch = stylized_pitch[i] - stylized_pitch[i - 1]
        delta_time = stylized_time[i] - stylized_time[i - 1]
        slope = delta_pitch / delta_time if delta_time != 0 else 0
        slopes.append(slope)
    return np.array(slopes)

def interpolate_wav_data(duration, data_points, time_steps):
    num_time_steps = int(float(duration[:-1]) * 50)
    new_time_steps = np.linspace(0, 100, num_time_steps)
    interpolate_function = interpolate.interp1d(time_steps, data_points, kind='linear', fill_value="extrapolate")
    new_data_points = interpolate_function(new_time_steps)
    return new_time_steps, new_data_points

def normalize(l):
    mi = np.min(l)
    ma = np.max(l)
    return (l - mi) / (ma - mi) * 100

def pitch_all(audio_file, duration, threshold):
    utterance_contour = get_pitch_over_threshold(audio_file, threshold)
    utterance_movement = extract_pitch_features_with_stylization_and_distance(audio_file, threshold)
    stylized_time, stylized_pitch = stylize_pitch_tier_from_sound(audio_file, frequency_resolution=2.0)
    slopes = calculate_slopes(stylized_time, stylized_pitch)
    v1 = normalize([i["pitch"] for i in utterance_contour])
    t1 = normalize([i["time"] for i in utterance_contour])
    v2 = normalize([i["slope"] for i in utterance_movement])
    t2 = normalize([i["end_time"] for i in utterance_movement])
    v3 = normalize(slopes)
    t3 = normalize(stylized_time[1:])
    t1, v1 = interpolate_wav_data(duration, v1, t1)
    t2, v2 = interpolate_wav_data(duration, v2, t2)
    t3, v3 = interpolate_wav_data(duration, v3, t3)
    return v1, v2, v3

### energy 추출 및 정규화 (calculate_energy(), get_normalized_RMS_energies() get_normalized_energies_per_intensities())
def calculate_energy(audio_file_or_y, sr=None, freq_floor=50, freq_ceiling=500):  
    if isinstance(audio_file_or_y, str):
        y, sr = librosa.load(audio_file_or_y, sr=None)
    elif isinstance(audio_file_or_y, np.ndarray):
        if sr is None:
            raise ValueError("Sample rate (`sr`) must be provided when using a preloaded signal.")
        y = audio_file_or_y
    else:
        raise TypeError("Invalid input type. Expected a file path (str) or preloaded signal (np.ndarray).")
    S = librosa.stft(y)  # .stft(y, n_fft=2048, hop_length=512) 지정 필요함 
    S_magnitude = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr)
    freq_mask = (freqs >= freq_floor) & (freqs <= freq_ceiling)
    S_magnitude = S_magnitude[freq_mask, :]
    energies = np.sqrt(np.sum(S_magnitude**2, axis=0))
    return energies

# 해당 코드 추가함 : RMS 기반 에너지 정규화 코드 
def get_normalized_RMS_energies(energies, sr=None, freq_floor=50, freq_ceiling=500):
    normalized_energies = energies / np.linalg.norm(energies) # RMS 에너지 기준으로 정규화
    return normalized_energies

def get_normalized_energies_per_intensities(audio_file, freq_floor=50, freq_ceiling=500):  # intensity와 비교(시각화) 위한 energy 범위 정규화 코드 
    y, sr = librosa.load(audio_file, sr=None)
    energies = calculate_energy(audio_file, freq_floor=freq_floor, freq_ceiling=freq_ceiling)  # energies = energy(y)
    sound = parselmouth.Sound(audio_file)
    intensity_obj = sound.to_intensity(minimum_pitch=50.0)
    intensity_times = intensity_obj.xs()  
    intensities = intensity_obj.values.flatten() 
    intensities = np.clip(intensities, 0, None) 
    get_normalized_energies_per_intensities = energies / np.max(energies) * np.max(intensities)
    return get_normalized_energies_per_intensities

full_data = pd.read_csv(drive_csv_path)
full_data = full_data.sample(frac=1).reset_index(drop=True)
results = []
storage_client = storage.Client(project=project_id)
b = storage_client.get_bucket(bucket_name)

def process_row(args):
    index, row, bucket, threshold = args
    file_name = row["voice_piece_filename"]
    audio_file = f"gs://{bucket.name}/data/{file_name}"
    blob = bucket.blob("data/" + file_name)
    blob.download_to_filename(os.path.join("/root/data", file_name))
    try:
        v1, v2, v3 = pitch_all(os.path.join("/root/data", file_name), row["voice_piece_duration"], threshold)
        print(index)
        return {
            'index': index,
            'voice_piece_filename': row["voice_piece_filename"],
            'v1': v1.tolist(),
            'v2': v2.tolist(),
            'v3': v3.tolist()
        }
    except Exception as e:
        print(f"Err")
        return {
            'index': index,
            'voice_piece_filename': row["voice_piece_filename"],
            'v1': None,
            'v2': None,
            'v3': None
        }

def preprocess(threshold=10):
    global results
    args = [(i, full_data.iloc[i], b, threshold) for i in range(51000)]
    with ThreadPoolExecutor() as executor:
        for idx, result in enumerate(executor.map(process_row, args)):
            results.append(result)
            if (idx + 1) % 1000 == 0:
                temp_data = pd.DataFrame(results)
                temp_data.to_csv("/root/temp.csv", index=False)
                print(f'{idx + 1}th data processed and saved as CSV')

preprocess()