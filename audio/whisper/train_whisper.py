# train #
import os
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
import gc
import jiwer
import pyctcdecode
import kenlm
import evaluate
import whisper_model
import whisper_dataset



class config:
    metrics = ["wer"] # metrics, 넣으면 추가됨
    output_dir = "/content/drive/MyDrive/audio/models/output_2" # model 어따가 save 할지 
    model_name_or_path = "sawradip/bengali-whisper-medium-tugstugi" #사용할 모델 경로 or hugging face 이름 
    use_early_stopping_callback = True # 얼리 스토핑 사용 여부 
    early_stopping_patience = 4 # 몇번 참을지
    save_base_model = False # 처음 로드한 모델 base 폴더에 저장할지 여부

    audio_id_col = "id" # audio의 id(구별자)가 들어있는 column
    target_col = "normalized" # target label이 들어있는 column
    original_sampling_rate = 32000 # 원본 오디오 sr
    sampling_rate = 16000 # target sr
    audio_absolute_path = '/content/drive/MyDrive/audio/dataset/train_mp3s/{}.mp3' # audio 절대경로 format
    max_word = 100 # label 최대 단어 갯수 제한
    max_frame = 3000 # mel_spectogram의 frame 길이 제한
    use_multiple_padding = None # 양옆으로 패딩할지 안할지
    pad_token = -100 # 패드의 토큰 id 
CFG = config()


training_args = TrainingArguments(
    output_dir=CFG.output_dir,
    overwrite_output_dir=True,
    group_by_length=False,
    optim = "adamw_torch",
    lr_scheduler_type='cosine', # 바꿀때 참고 https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/optimizer_schedules#transformers.SchedulerType
    weight_decay=0.01, 
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    # save_strategy="steps",
    num_train_epochs=1,
    fp16=True,
    # save_steps=100,
    eval_steps=100,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=350,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    prediction_loss_only=False,
    auto_find_batch_size=False,
    report_to="none", # wandb같은거 사용시 건들여주기
)


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f"Alert: Directory {directory} already exists -> pretrained model could be overwritten")
createDirectory(CFG.output_dir)
createDirectory(CFG.output_dir + "/base")



# set metric for the validation and save model
metrics = [evaluate.load(name) for name in CFG.metrics]

def get_comput_metrics_func(processor):
    processor = processor
    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        results_dict = {k: v.compute(predictions=pred_str, references=label_str) for (k, v) in zip(config.metrics, metrics)}
        return results_dict
    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


class whisper_asr:
    def __init__(self, train_df, valid_df):
        print("loading model...")
        self.model, self.processor =  whisper_model.load_model(CFG.model_name_or_path)
        
        self.callbacks = []
        self.train_dataset = None
        self.valid_dataset = None

        print("loading dataset...")
        if train_df is not None:
            self.train_dataset = whisper_dataset.WhisperDataset(train_df, self.processor, CFG.audio_id_col, CFG.target_col, CFG.original_sampling_rate, CFG.sampling_rate, CFG.audio_absolute_path)
        if valid_df is not None:
            self.valid_dataset = whisper_dataset.WhisperDataset(valid_df, self.processor, CFG.audio_id_col, CFG.target_col, CFG.original_sampling_rate, CFG.sampling_rate, CFG.audio_absolute_path)
        self.data_collator = whisper_dataset.DataCollatorCTCWithPadding(self.processor, CFG.pad_token, True, CFG.max_frame, CFG.max_word, CFG.use_multiple_padding, CFG.use_multiple_padding)
        print("Done!")

        

        if CFG.save_base_model:
            self.save_model(CFG.output_dir + "/base")
        if CFG.use_early_stopping_callback:
            self.callbacks.append(EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience))
        

    def save_model(self, directory):
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
    
    def train(self):
        print("training... !!!!!!")
        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=get_comput_metrics_func(self.processor),
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.processor.feature_extractor,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        # 마지막 결과 자동 저장
        trainer.save_model(CFG.output_dir)
        self.processor.save_pretrained(CFG.output_dir)
        return trainer


# load pretrained_model and processor from huggingface
# model, processor = whisper_model.load_model(CFG.model_name_or_path)
# if CFG.save_base_model:
#     save_model(CFG.output_dir + "/base", model, processor)


# def train(train_df = None, valid_df = None):
    # callbacks = []
    # train_dataset = None
    # valid_dataset = None

    # print("loading dataset...")
    # if train_df is not None:
    #     train_dataset = whisper_dataset.WhisperDataset(train_df, processor, CFG.audio_id_col, CFG.target_col, CFG.original_sampling_rate, CFG.sampling_rate, CFG.audio_absolute_path)
    # if valid_df is not None:
    #     valid_dataset = whisper_dataset.WhisperDataset(valid_df, processor, CFG.audio_id_col, CFG.target_col, CFG.original_sampling_rate, CFG.sampling_rate, CFG.audio_absolute_path)
    # data_collator = whisper_dataset.DataCollatorCTCWithPadding(processor, CFG.pad_token, True, CFG.max_frame, CFG.max_word, CFG.use_multiple_padding, CFG.use_multiple_padding)
    # print("Done!")
    
    # if CFG.use_early_stopping_callback:
    #     callbacks.append(EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience))


    # print("training... !!!!!!")
    # trainer = Trainer(
    #     model=model,
    #     data_collator=data_collator,
    #     args=training_args,
    #     compute_metrics=compute_metrics,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     tokenizer=processor.feature_extractor,
    #     preprocess_logits_for_metrics = preprocess_logits_for_metrics,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # )

    # trainer.train()

    # # 마지막 결과 자동 저장
    # trainer.save_model(CFG.output_dir)
    # processor.save_pretrained(CFG.output_dir)
    # return trainer
