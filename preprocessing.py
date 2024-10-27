import argparse
import pandas as pd
import os
import librosa
from datasets import load_dataset, DatasetDict
import whisperx
import numpy as np
from num2words import num2words
import torch


from utils.get_delivery import delivery_feat_preprocess

max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000


def argsparser():
    parser = argparse.ArgumentParser(description="Preprocess the dataset")
    parser.add_argument("--input", type=str, help="Path to the input dataset")
    # parser.add_argument("--output", type=str, help="Path to the output dataset")
    return parser.parse_args()


def num_to_words(text):
    """
    Convert Numbers to Words

    Args:
        text (str): String to which the function is to be applied, string

    Returns:
        Clean string with converted Numbers to Words
    """
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index])
        elif after_spliting[index][:-1].isdigit():  # "123." -> "123"
            after_spliting[index] = (
                num2words(after_spliting[index][:-1]) + after_spliting[index][-1]
            )
        elif (
            after_spliting[index].replace(",", "").replace("$", "").isdigit()
        ):  # "1,000" -> "1000"
            after_spliting[index] = num2words(
                after_spliting[index].replace(",", "").replace("$", "")
            )
    numbers_to_words = " ".join(after_spliting)
    return numbers_to_words


def convert_numbers_to_words(examples):
    examples["asr"] = [num_to_words(text) for text in examples["asr"]]
    return examples


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(
        speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate
    )
    return mono_waveform, sampling_rate


def force_alignment(examples):
    examples["word_segments"] = []
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")

    durations = [
        librosa.get_duration(y=np.array(audio), sr=sampling_rate)
        for audio, sampling_rate in zip(examples["audio"], examples["sampling_rate"])
    ]
    segments = [
        [{"text": text, "start": 0, "end": duration}]
        for text, duration in zip(examples["asr"], durations)
    ]

    for segment, audio in zip(segments, examples["audio"]):
        audio_tensor = torch.from_numpy(np.array(audio)).float().to("cuda")
        result = whisperx.align(
            segment,
            model_a,
            metadata,
            audio_tensor,
            device="cuda",
            return_char_alignments=False,
        )
        examples["word_segments"].append(result["word_segments"])

    return examples


def load_audio_files(examples):
    audio_list = []
    sampling_rate_list = []

    for wav_path in examples["wav_path"]:
        audio, sampling_rate = speech_file_to_array_fn(wav_path)
        audio_list.append(np.array(audio, dtype=np.float32))
        sampling_rate_list.append(sampling_rate)

    # Convert lists to NumPy arrays after the loop
    examples["audio"] = audio_list
    examples["sampling_rate"] = np.array(
        sampling_rate_list, dtype=np.int64
    )  # Ensure correct dtype

    return examples


def load_LTTC_dataset(input_path: str):
    """
    This function loads the data from the origin csv file and the questions metadata csv file.
    Specifically designed for LTTC answering-questions dataset.
    """

    # Check if the input path exists
    if not os.path.exists(f"{input_path}/dev"):
        print(f"Input path {input_path}/dev does not exist")
        return
    if not os.path.exists(f"{input_path}/train"):
        print(f"Input path {input_path}/train does not exist")
        return

    data_files = {}
    metadata_files = {}

    for folder in ["dev", "train", "testseen"]:
        # check if dev and train folder exists
        if not os.path.exists(f"{input_path}/{folder}"):
            print(f"Input path {input_path}/{folder} does not exist")
            return

        files = os.listdir(f"{input_path}/{folder}")
        if len(files) == 0:
            print(f"Input path {input_path}/{folder} is empty")
            return

        # Get the origin csv file
        origin_csv_path = None
        for file in files:
            if file.endswith("_base.csv") or file.endswith("_expand.csv"):
                origin_csv_path = f"{input_path}/{folder}/{file}"
                break

        if origin_csv_path is None:
            print(f"Origin csv file not found in {input_path}/{folder}")
            return

        # Get all the questions metadata csv files
        questions_metadata_csv_path = []
        for file in files:
            if file.endswith("_question.csv"):
                questions_metadata_csv_path.append(f"{input_path}/{folder}/{file}")

        def get_question_set_id(path):
            return path.split("/")[-1].split("_")[0]

        data_files[folder] = origin_csv_path
        metadata_files[folder] = {
            get_question_set_id(path): path for path in questions_metadata_csv_path
        }

    raw_datasets = load_dataset("csv", data_files=data_files)
    raw_metadata = {
        folder: {key: pd.read_csv(path) for key, path in value.items()}
        for folder, value in metadata_files.items()
    }

    return raw_datasets, raw_metadata


def main(args):
    input_path = args.input
    input_path = os.path.abspath(input_path)

    # Load the dataset
    raw_datasets, raw_metadata = load_LTTC_dataset(input_path)

    ## Load the audio files
    print("---Loading audio files---")
    raw_datasets = raw_datasets.map(load_audio_files, batched=True, num_proc=8)

    ## Transform numbers to words
    print("---Transform numbers to words---")
    raw_datasets = raw_datasets.map(convert_numbers_to_words, batched=True, num_proc=8)

    ## Force alignment
    print("---Force alignment---")
    raw_datasets = raw_datasets.map(force_alignment, batched=True, num_proc=2)

    ## Delivery feature
    print("---Delivery feature---")
    raw_datasets = raw_datasets.map(
        delivery_feat_preprocess, remove_columns=["word_segments"], num_proc=8
    )

    raw_datasets = raw_datasets.remove_columns(["audio"])

    if type(raw_datasets) == DatasetDict:
        for k, dataset in raw_datasets.items():
            dataset.to_csv(f"features/{k}.csv")
    else:
        raw_datasets.to_csv(f"features/output.csv")


if __name__ == "__main__":
    args = argsparser()
    main(args)
