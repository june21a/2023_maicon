# model.py #
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# load pretrained model
def load_model(model_name_or_path = "sadidul012/whisper-small-bengali", freeze_encoder = True):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    if freeze_encoder:
        model.freeze_encoder()
    return model, processor