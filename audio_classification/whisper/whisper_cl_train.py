import os
import torch
from transformers import  AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, pipeline
import evaluate
import whisper_cl_model
import whisper_cl_dataset



class config:
    sampling_rate = 16000
    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * sampling_rate
    # 위에 3개는 모델 config 잘 안보고 바꾸면 오류납니다, 모델 config.json하고 processor_config.json 보고 바꿔주세요

    activation_dropout = 0.2
    attention_dropout = 0.2
    dropout = 0.2

    train_split_rate = "[0%:50%]"
    test_size = 0.2
    data_dir = "/content/drive/MyDrive/audio_classification/dataset/train"
    top_dropout = 0.3

    metric = "accuracy"
    output_dir = "/content/drive/MyDrive/audio_classification/models/output_1" # model 어따가 save 할지 
    model_name_or_path = "sanchit-gandhi/whisper-medium-fleurs-lang-id" #사용할 모델 경로 or hugging face 이름 
    use_early_stopping_callback = True # 얼리 스토핑 사용 여부 
    early_stopping_patience = 4 # 몇번 참을지
    save_base_model = True # 처음 로드한 모델 base 폴더에 저장할지 여부
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
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    num_train_epochs=1,
    fp16=True,
    save_steps=20,
    eval_steps=20,
    logging_steps=20,
    learning_rate=1e-3,
    warmup_steps=60,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model=CFG.metric,
    greater_is_better=True,
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
metric = evaluate.load(CFG.metric)

def compute_metrics(eval_pred):
    return metric.compute(predictions=eval_pred.predictions[0], references=eval_pred.label_ids)

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels




class whisper_cl:
    def __init__(self):
        print("loading dataset and model...")
        self.train_ds = whisper_cl_dataset.load_dataset_from_directory(data_dir = CFG.data_dir, train_split_rate = CFG.train_split_rate, test_size = CFG.test_size)
        self.feature_extractor = whisper_cl_model.load_processor(CFG.model_name_or_path)
        self.labels = whisper_cl_model.get_label_from_ds(self.train_ds)
        self.train_ds = self.train_ds.map(whisper_cl_dataset.get_prepare_dataset_func(self.feature_extractor))
        self.model = whisper_cl_model.load_model(self.labels, CFG.model_name_or_path, True)
        self.callbacks = []

        self.model.config.dropout = CFG.dropout
        self.model.config.attention_dropout = CFG.attention_dropout
        self.model.config.activation_dropout = CFG.activation_dropout
        if CFG.save_base_model:
            print("saving base model...")
            self.save_model(CFG.output_dir + "/base", self.model, self.feature_extractor)
        
        if CFG.use_early_stopping_callback:
            self.callbacks.append(EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience))

    def save_model(self, directory, model, feature_extractor):
        model.save_pretrained(directory)
        feature_extractor.save_pretrained(directory)
    
    def train(self):
        print("training... !!!!!!")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=self.train_ds["train"],
            eval_dataset=self.train_ds["test"],
            tokenizer=self.feature_extractor,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
            callbacks=self.callbacks,
        )

        trainer.train()

        # 마지막 결과 자동 저장
        trainer.save_model(CFG.output_dir)
        self.feature_extractor.save_pretrained(CFG.output_dir)
        return trainer


# load pretrained_model and processor from huggingface
print("loading dataset...")
train_ds = whisper_cl_dataset.load_dataset_from_directory(data_dir = CFG.data_dir, train_split_rate = CFG.train_split_rate, test_size = CFG.test_size)
encoded_ds = train_ds.map(whisper_cl_dataset.prepare_dataset)
labels = whisper_cl_model.get_label_from_ds(train_ds)
model, feature_extractor = whisper_cl_model.load_model(labels, CFG.model_name_or_path, True)
if CFG.save_base_model:
    save_model(CFG.output_dir + "/base", model, feature_extractor)




def train():
    callbacks = []
    if CFG.use_early_stopping_callback:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience))

    
    print("training... !!!!!!")
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["test"],
        tokenizer=feature_extractor,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # 마지막 결과 자동 저장
    trainer.save_model(CFG.output_dir)
    feature_extractor.save_pretrained(CFG.output_dir)
    return trainer