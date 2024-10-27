
# =======================Preprocessing===============================
import pandas as pd
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spacy
from transformer_model import Transformer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Processor
# from transformers import WhisperProcessor, WhisperModel
import os
import re
import warnings
warnings.filterwarnings('ignore')
from text_preprocessing import *
import json

def phi(task_label, true_label):
    if task_label == true_label:
        return 0
    elif abs(task_label - true_label) == 1:
        return args.softlabel_para
    else:
        return float("inf")   

def func_softLabels(label):
    exp_terms = torch.exp(-torch.tensor([phi(i, label) for i in range(num_labels)]))
    soft_label = []
    for i in range(num_labels):
        k = torch.exp(-torch.tensor(phi(i, label))) / exp_terms.sum()
        soft_label.append(k.item())

    return soft_label

import librosa
max_audio_length = 16000 * 90  # 假設音訊的最大長度是16000個樣本點
target_sampling_rate = 16000
def speech_file_to_array_fn(path):
    speech_array, sampling_rate = librosa.load(path, sr=target_sampling_rate)
    mono_waveform = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sampling_rate)

    return mono_waveform


def cal_parameter():
    total_params = sum(
        param.numel() for param in model.parameters()
    )
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f'Total: {total_params}')
    print(f'Trainable: {trainable_params}')

def preprocess(examples):

    target_list = [label_to_id(label, label_list) for label in examples[args.score]]
    
    # Softmax 
    # target_list = [func_softLabels(label_to_id(label, label_list)) for label in examples[args.score]]

    vec_list = [json.loads(item) for item in examples['delivery_vec']]
    vec = pad_delivery_sequences(vec_list)

    # ================== Language =================================
    onehot_encoding = [text2BinVec(txt) for txt in examples[args.asr_transcript]]
    padded_seq = pad_sequence(onehot_encoding) # [batch, sentence_length, 247]
    masked_seq = mask_seq(onehot_encoding)

    # ==================Content=================================
    padded_response_emb = [combineContext(res) for res in examples[args.asr_transcript]]
    padded_prompt_emb   = [combineContext(prompt) for prompt in examples[args.description]]

    result = {
        "padded_seq": padded_seq,
        "masked_seq": masked_seq,
        "delivery_tra": vec,
    }
    print("**************************************")

    # BERT
    response_tokens = tokenizer(padded_response_emb, 
                                max_length=256,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')

    prompt_tokens = tokenizer(padded_prompt_emb, 
                                max_length=256,
                                truncation=True,
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')
    
    # ========================= Wav2vec Delivery =================================
    # speech_list = [speech_file_to_array_fn(path) for path in examples[args.path_column]]
    # result['delivery_emb'] = processor(speech_list, 
    #                                 sampling_rate=target_sampling_rate,
    #                                 max_length=max_audio_length,
    #                                 truncation=True,
    #                                 padding='max_length',
    #                                 return_tensors="pt"
    #                             )['input_values']


    result['padded_res_emb'] = response_tokens['input_ids']
    result['attention_mask_res'] = response_tokens['attention_mask']

    result['padded_pro_emb'] = prompt_tokens['input_ids']
    result['attention_mask_pro'] = prompt_tokens['attention_mask']

    result['labels'] = target_list

    return result

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    acc = accuracy_score(eval_pred.label_ids, predictions)
    return {"accuracy": float(acc)}

# soft label
# from sklearn.metrics import accuracy_score
# def compute_metrics(eval_pred):
#     predictions = np.argmax(eval_pred.predictions, axis=1)
#     references = np.argmax(eval_pred.label_ids, axis=1) # softlabel才需這樣算！
#     acc = accuracy_score(references, predictions)
    # return {"accuracy": float(acc)}
    
np.random.seed(42)
def random_round(scores):
    round_up = np.random.rand(*scores.shape) < 0.5 
    return np.where(round_up, np.ceil(scores), np.floor(scores))


import math
def dataset_prepare(file_path):
    df = pd.read_csv(file_path)
    # random score
    # df[args.score] = random_round(df[args.score])
    # baseline
    df[args.score] = df[args.score].apply(math.floor)     # 3.5 -> 3.0
    df = Dataset.from_dict(df)
    return df

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_id", type=int, default=0, help="cuda device")
    parser.add_argument('--train_file',     help='path')
    parser.add_argument('--dev_file',       help='path')
    parser.add_argument('--output_dir',     help='path')
    # parser.add_argument('--asr_transcript',    default='asr_transcripttion', help='text_value')
    parser.add_argument('--asr_transcript', default='whisperX_transcription', help='text_value')

    parser.add_argument('--description',    default='prompt', help='description')
    parser.add_argument('--score',   default='grade', help='score_value')
    parser.add_argument('--path_column',    default='wav_path', help='path_value')
    parser.add_argument('--train_epochs',   type=int, default=8, help='epochs')
    parser.add_argument('--train_batch',    type=int, default=2, help='epochs')
    parser.add_argument('--eval_batch',     type=int, default=2, help='batch')
    parser.add_argument('--grad_acc',       type=int, default=4, help='grad_acc')
    parser.add_argument('--learning_rate',  type=float, default=1e-3, help='grad_acc')
    parser.add_argument('--softlabel_para', type=float, default=0.8, help='score_value')

    args = parser.parse_args()
    # ------------------------Cuda Device------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, str(args.cuda_id))

    # --------------------------- Loading Data------------------------------------
    train_df = dataset_prepare(args.train_file)
    dev_df = dataset_prepare(args.dev_file)

    label_list2 = train_df.unique(args.score)
    label_list = [1, 2, 3, 4, 5] 
    label_list.sort()
    label_list2.sort()
    num_labels = len(label_list)
    print('Actually label list', label_list2)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    # --------------------------- BERT FeatureExtractor ------------------------------------
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    # --------------------------Data Preparation---------------------------
    train_df = train_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcript', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcripttion', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI 
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
    )
    dev_df = dev_df.map(
        preprocess,
        batch_size=100,
        batched=True,
        # remove_columns=['speaker_id', 'score', 'classification', 'description', 'example', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)', 'asr_transcript', 'wav_path']
        # remove_columns=['speaker_id', 'form_id', 'grade', 'wav_path', 'asr_transcripttion', 'prompt', 'confidence_score', 'acoustic_score', 'lm_score', 'pitch', 'intensity', 'pause(sec)', 'silence(sec)', 'duration(sec)']
        # WhisperX
        remove_columns=['example', 'wpm', 'total_num_word', 'level_1', 'level_2', 'level_3', 'seq_len', 'key1', 'key2', 'key3', 'key4', 'avg', 'all_avg_score', 'Threshold_Count', 'mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'mean_long_silence', 'mean_silence', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']
        # HI
        # remove_columns=['mean_pitch', 'mean_intensity', 'duration', 'localJitter', 'localShimmer', 'rapJitter', 'long_silence', 'silence', 'long_silence_num', 'silence_num', 'std_energy', 'avg_spectral', 'avg_energy_entropy', 'zero_cross_num', 'v_to_uv_ratio', 'voice_count', 'unvoice_count', 'more3word', 'num_word', 'whisperX_transcription', 'delivery_vec']

    )
    print('train_df', train_df)
    print("dev_df", dev_df)

    # ------------------------Model------------------------------
    from multi_subModule import *
    
    model = MultiFeatureModel(device, num_labels=num_labels)
    model.to(device)
    cal_parameter()

    # ======================= Trainer ===================================
    from transformers import TrainingArguments, Trainer
    import wandb
    wandb.init(mode="disabled")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch,   # 32
        per_device_eval_batch_size=args.eval_batch,     # 32
        gradient_accumulation_steps=args.grad_acc,
        evaluation_strategy="steps",
        save_steps=10,
        eval_steps=10,
        num_train_epochs=args.train_epochs,
        logging_steps=8,
        learning_rate=args.learning_rate,
        save_total_limit=5, # 限制保存的模型检查点的总数，即限制保存模型的历史版本数量
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_df,
        eval_dataset=dev_df,
    )
    trainer.train()
