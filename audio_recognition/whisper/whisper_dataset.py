import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as tat
import transformers
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

class config:
    audio_id_col = "id" # audio의 id(구별자)가 들어있는 column
    target_col = "normalized" # target label이 들어있는 column
    original_sampling_rate = 32000 # 원본 오디오 sr
    sampling_rate = 16000 # target sr
    audio_absolute_path = '/content/drive/MyDrive/audio/dataset/train_mp3s/{}.mp3' # audio 절대경로 format
    max_word = 100 # label 최대 단어 갯수 제한
    max_frame = 4000 # mel_spectogram의 frame 길이 제한
    use_multiple_padding = None # 양옆으로 패딩할지 안할지
    pad_token = -100 # 패드의 토큰 id 
CFG = config()


def get_vocab_dict(processor):
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    return sorted_vocab_dict


# df : pd.DataFrame   /  processor : transformers.AutoProcessor
class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, audio_id_col, target_col, original_sampling_rate, sampling_rate, audio_absolute_path):
        self.df = df
        self.audio_id_col = audio_id_col
        self.processor = processor 
        self.sentences = df[target_col].values
        self.resampler = tat.Resample(original_sampling_rate, sampling_rate)
        self.model_input_name = processor.model_input_names[0]
        self.audio_absolute_path = audio_absolute_path
        self.sampling_rate = sampling_rate

    # batch = {"input_features" : mel_spectogram, "labels": tokenized sentence}
    def __getitem__(self, idx):
        apath = self.audio_absolute_path
        waveform, sample_rate = torchaudio.load(apath.format(self.df.loc[idx, self.audio_id_col]), format="mp3") # 로드
        waveform = self.resampler(waveform) #리샘플링

        
        ###############audio augmentation  여기에 넣기###############
        #####################################################################


        # hugging face format으로 만들어주기
        batch = dict()
        y = self.processor(waveform.reshape(-1), sampling_rate=self.sampling_rate).input_features[0] # 여기서 audio -> mel_spectogram
        batch[self.model_input_name] = y
        batch["labels"] = self.processor(text=self.sentences[idx], sampling_rate=self.sampling_rate).input_ids # 여기서 label 토큰화
        return batch

    def __len__(self):
        return len(self.df)


@dataclass
class DataCollatorCTCWithPadding:
    processor: Optional[transformers.AutoProcessor]
    pad_token: Optional[int] = CFG.pad_token
    padding: Union[bool, str] = True
    max_length: Optional[int] = CFG.max_frame # max_frame, sample_rate * 시간(second) = num_frame
    max_length_labels: Optional[int] = CFG.max_word # 100단어로 짜르기 # label 최대 단어 갯수
    pad_to_multiple_of: Optional[int] = CFG.use_multiple_padding # 패딩이 양옆으로 됨 -> 패딩 2배
    pad_to_multiple_of_labels: Optional[int] = CFG.use_multiple_padding

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        
        # dictionary refectoring for training
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        
        # padding
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", 
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), self.pad_token)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch