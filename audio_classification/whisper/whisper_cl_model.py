import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

class config:
    top_dropout = 0.3
    model_name_or_path = "sanchit-gandhi/whisper-base-ft-common-language-id"
CFG = config()

def get_label_from_ds(ds):
    return ds["train"].features["label"].names

def label_to_dict(labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    print(f"all_label_num : {len(label2id)}\n\n {labels}")
    return label2id, id2label

def load_processor(model_name_or_path = CFG.model_name_or_path):
    return AutoFeatureExtractor.from_pretrained(model_name_or_path)

# load classifier model
def load_model(labels, model_name_or_path = "sanchit-gandhi/whisper-base-ft-common-language-id", freeze_encoder = True):
    label2id, id2label = label_to_dict(labels)
    model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, num_labels = len(labels), label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
    model.config.num_labels = len(labels)
    if freeze_encoder:
        model.freeze_encoder()
    return model


def custom_model(labels, model_name_or_path = CFG.model_name_or_path, freeze_encoder = True):
    model = AutoModelForAudioClassification.from_pretrained(model_name_or_path)
    processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

    model.projector = nn.Sequential(
        (nn.Linear(512, 256)),
        nn.Dropout(CFG.top_dropout),
        nn.ReLU(),
    )

    model.classifier = nn.Sequential(
        (nn.Linear(256, len(labels))),
        nn.Softmax()
    )

    model.config.num_labels = len(labels)
    return model, processor
