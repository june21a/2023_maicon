# train #
import os
import pandas as pd
import yaml
import argparse
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
import gc
import jiwer
import evaluate
import whisper_model
import whisper_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, default='../../config/recognition_setting.yml', help='사용할 config 경로')
    parser.add_argument('train_csv_path', type=str, required=True, help="train.csv의 경로")
    parser.add_argument('val_csv_path', type=str, required=True, help="val.csv의 경로")
    
    args = parser.parse_args()
    return args
    

def load_yaml(yml_path):
    # YAML 파일을 읽어서 Python 딕셔너리로 변환
    with open(yml_path, 'r') as yml_file:
        yml_data = yaml.safe_load(yml_file)
    return yml_data


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f"Alert: Directory {directory} already exists -> pretrained model could be overwritten")


def get_compute_metrics_func(processor, metric_names):
    processor = processor
    metrics = [evaluate.load(name) for name in metric_names]
    
    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        results_dict = {k: v.compute(predictions=pred_str, references=label_str) for (k, v) in zip(metric_names, metrics)}
        return results_dict
    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


class WhisperAsr:
    def __init__(self, train_df, valid_df, CFG):
        print("loading model...")
        self.model, self.processor =  whisper_model.load_model(CFG["model_name_or_path"])
        
        self.callbacks = []
        self.train_dataset = None
        self.valid_dataset = None
        self.CFG = CFG

        print("loading dataset...")
        if train_df is not None:
            self.train_dataset = whisper_dataset.WhisperDataset(
                train_df, 
                self.processor, 
                self.CFG['audio_id_col'], 
                self.CFG['target_col'], 
                self.CFG['original_sampling_rate'], 
                self.CFG['sampling_rate'], 
                self.CFG['audio_absolute_path']
            )

        if valid_df is not None:
            self.valid_dataset = whisper_dataset.WhisperDataset(
                valid_df, 
                self.processor, 
                self.CFG['audio_id_col'], 
                self.CFG['target_col'], 
                self.CFG['original_sampling_rate'], 
                self.CFG['sampling_rate'], 
                self.CFG['audio_absolute_path']
            )

        self.data_collator = whisper_dataset.DataCollatorCTCWithPadding(
            self.processor, 
            self.CFG['pad_token'], 
            True, 
            self.CFG['max_frame'], 
            self.CFG['max_word'], 
            self.CFG['use_multiple_padding'], 
            self.CFG['use_multiple_padding']
        )
        print("Done!")

        createDirectory(self.CFG["output_dir"])
        createDirectory(self.CFG["output_dir"] + "/base")

        if self.CFG["save_base_model"]:
            self.save_model(self.CFG["output_dir"] + "/base")
        if self.CFG["use_early_stopping_callback"]:
            self.callbacks.append(EarlyStoppingCallback(early_stopping_patience=self.CFG["early_stopping_patience"]))
        

    def save_model(self, directory):
        self.model.save_pretrained(directory)
        self.processor.save_pretrained(directory)
    
    def train(self, training_args):
        print("training... !!!!!!")
        trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=get_compute_metrics_func(self.processor),
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset,
            tokenizer=self.processor.feature_extractor,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()

        # 마지막 결과 자동 저장
        trainer.save_model(self.CFG["output_dir"])
        self.processor.save_pretrained(self.CFG["output_dir"])
        return trainer


def main():
    args = parse_arguments()
    train_df = pd.read_csv(args.train_csv_path)
    val_df = pd.read_csv(args.val_csv_path)
    CFG = load_yaml(args.cfg_path)
    
    
    training_args = TrainingArguments(
        output_dir=CFG['output_dir'],  # 모델 저장 디렉토리
        overwrite_output_dir=CFG['overwrite_output_dir'],  # 기존 output_dir 덮어쓸지 여부
        group_by_length=CFG['group_by_length'],  # 데이터 샘플들을 길이에 따라 그룹화할지 여부
        optim=CFG['optim'],  # 옵티마이저 (AdamW 사용)
        lr_scheduler_type=CFG['lr_scheduler_type'],  # 학습률 스케줄러 종류 (cosine)
        weight_decay=CFG['weight_decay'],  # 가중치 감소
        adam_beta1=CFG['adam_beta1'],  # Adam 옵티마이저의 beta1 파라미터
        adam_beta2=CFG['adam_beta2'],  # Adam 옵티마이저의 beta2 파라미터
        per_device_train_batch_size=CFG['per_device_train_batch_size'],  # 훈련 배치 크기
        per_device_eval_batch_size=CFG['per_device_eval_batch_size'],  # 평가 배치 크기
        gradient_accumulation_steps=CFG['gradient_accumulation_steps'],  # 그래디언트 누적 스텝
        evaluation_strategy=CFG['evaluation_strategy'],  # 평가 전략 (steps 기준)
        save_strategy=CFG['save_strategy'],  # 저장 전략 (steps 기준)
        num_train_epochs=CFG['num_train_epochs'],  # 훈련 에폭 수
        fp16=CFG['fp16'],  # 16-bit 부동소수점 사용 여부
        save_steps=CFG['save_steps'],  # 저장 주기 (steps 기준)
        eval_steps=CFG['eval_steps'],  # 평가 주기 (steps 기준)
        logging_steps=CFG['logging_steps'],  # 로그 기록 주기
        learning_rate=CFG['learning_rate'],  # 초기 학습률
        warmup_steps=CFG['warmup_steps'],  # 학습률 워밍업 단계
        save_total_limit=CFG['save_total_limit'],  # 저장할 모델 수 제한
        load_best_model_at_end=CFG['load_best_model_at_end'],  # 훈련 종료 후 가장 좋은 모델 로드 여부
        metric_for_best_model=CFG['metric_for_best_model'],  # 성능 평가 지표 (이 경우 'wer')
        greater_is_better=CFG['greater_is_better'],  # 지표에서 "클수록 좋은지"
        prediction_loss_only=CFG['prediction_loss_only'],  # 예측 손실만 계산할지 여부
        auto_find_batch_size=CFG['auto_find_batch_size'],  # 자동 배치 크기 탐색 여부
    )
    
    whisper = WhisperAsr(train_df, val_df, CFG)
    whisper.train(training_args=training_args)


if __name__=="__main__":
    main()