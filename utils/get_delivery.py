import numpy as np
import json
import librosa
from pyAudioAnalysis import ShortTermFeatures
import parselmouth
import logging

max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(
        speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate
    )
    return mono_waveform, sampling_rate


def save_data(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=2)


def delivery_feat_preprocess(example):
    example["delivery_vector"] = []
    for word_segment in example["word_segments"]:
        try:
            start_sec = float(word_segment["start"])
            end_sec = float(word_segment["end"])

            duration = end_sec - start_sec

            if duration <= 0.07:
                tt = (0.08 - duration) / 2
                start_sec -= tt
                end_sec += tt

            duration = end_sec - start_sec

            start_sample = int(start_sec * example["sampling_rate"])
            end_sample = int(end_sec * example["sampling_rate"])

            # 截取指定时间段的音频数据
            segment = example["audio"][start_sample:end_sample]

            frame_size = int(0.050 * example["sampling_rate"])  # 50ms
            hop_size = int(0.025 * example["sampling_rate"])  # 25ms
            if duration <= 0.075:
                logging.warning(
                    f"Duration of the segment is too short for feature extraction: {duration}, should be at least 0.075s"
                )
                voiced_count = 0
                unvoiced_count = 0
                voiced_to_unvoiced_ratio = 0
                zero_crossing_rate = 0
                spectral_centroid = 0
                energy_entropy = 0
                std_dev_energy = 0
            else:
                F, feature_names = ShortTermFeatures.feature_extraction(
                    segment, example["sampling_rate"], frame_size, hop_size
                )
                # 提取特定特征
                zero_crossing_index = feature_names.index("zcr")
                energy_index = feature_names.index("energy")
                energy_entropy_index = feature_names.index("energy_entropy")
                spectral_centroid_index = feature_names.index("spectral_centroid")

                zcr = F[zero_crossing_index, :]
                zero_crossing_rate = np.mean(zcr)
                energy = F[energy_index, :]
                std_dev_energy = np.std(energy)
                energy_entropy = F[energy_entropy_index, :]
                spectral_centroid = F[spectral_centroid_index, :]

                energy_threshold = np.median(energy)  # Threshold for energy
                zcr_threshold = np.median(zcr)  # Threshold for zero-crossing rate

                # Classify frames as voiced or unvoiced
                voiced_frames = (energy > energy_threshold) & (zcr < zcr_threshold)
                unvoiced_frames = (energy <= energy_threshold) | (zcr >= zcr_threshold)

                # Calculate ratios
                voiced_count = np.sum(voiced_frames)
                unvoiced_count = np.sum(unvoiced_frames)
                voiced_to_unvoiced_ratio = (
                    voiced_count / unvoiced_count
                    if unvoiced_count != 0
                    else float("inf")
                )

            snd = parselmouth.Sound(example["wav_path"])
            snd_part = snd.extract_part(
                from_time=start_sec, to_time=end_sec, preserve_times=False
            )

            try:
                # pitch
                pitch = snd_part.to_pitch()
                pitch_values = pitch.selected_array["frequency"]
                pitch_values[pitch_values == 0] = np.nan
                mean_pitch = np.nanmean(pitch.selected_array["frequency"])
            except parselmouth.PraatError as e:
                mean_pitch = 0
                logging.warning(e)

            # intensity
            if 0.064 > duration:
                mean_intensity = 0
                logging.warning(
                    f"Duration of the segment is too short for intensity analysis: {duration}"
                )
            else:
                intensity = snd_part.to_intensity()
                mean_intensity = intensity.values.mean()

            # local Jitter, local Shimmer, rap jitter
            if mean_pitch == 0:
                localJitter = 0
                localShimmer = 0
                rapJitter = 0
            else:
                point_process = parselmouth.praat.call(
                    [snd_part, pitch], "To PointProcess (cc)"
                )
                localJitter = parselmouth.praat.call(
                    point_process,
                    "Get jitter (local)",
                    start_sec,
                    end_sec,
                    0.0001,
                    0.05,
                    1.3,
                )
                localShimmer = parselmouth.praat.call(
                    [snd, point_process],
                    "Get shimmer (local)",
                    start_sec,
                    end_sec,
                    0.0001,
                    0.05,
                    1.3,
                    1.6,
                )
                rapJitter = parselmouth.praat.call(
                    point_process,
                    "Get jitter (rap)",
                    start_sec,
                    end_sec,
                    0.0001,
                    0.05,
                    1.3,
                )

                if np.isnan(localJitter):
                    localJitter = 0
                if np.isnan(localShimmer):
                    localShimmer = 0
                if np.isnan(rapJitter):
                    rapJitter = 0
            
            delivery_features = {
                "duration": duration,
                "mean_pitch": mean_pitch,
                "mean_intensity": mean_intensity,
                "localJitter": localJitter,
                "localShimmer": localShimmer,
                "rapJitter": rapJitter,
                "std_energy": std_dev_energy,
                "avg_spectral": np.mean(energy_entropy),
                "avg_energy_entropy": np.mean(spectral_centroid),
                "zero_cross_rate": zero_crossing_rate,
                "v_to_uv_ratio": float(voiced_to_unvoiced_ratio),
                "voice_count": float(voiced_count),
                "unvoice_count": float(unvoiced_count),
            }
            example["delivery_vector"].append(list(delivery_features.values()))
        except Exception as e:
            print(f"wav_path: {example['wav_path']}")
            print(f"word_segment: {word_segment}")
            print(f"duration: {duration}")
            example["delivery_vector"].append([0] * 13)
            continue
    example["delivery_vector"] = np.array(example["delivery_vector"])
    return example
