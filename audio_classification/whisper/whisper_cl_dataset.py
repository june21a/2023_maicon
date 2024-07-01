from datasets import load_dataset, Audio
from whisper_cl_model import load_processor

class config:
    sampling_rate = 16000
    MAX_DURATION_IN_SECONDS = 30.0
    max_input_length = MAX_DURATION_IN_SECONDS * sampling_rate
    # 위에 3개는 모델 config 잘 안보고 바꾸면 오류납니다, 모델 config.json하고 processor_config.json 보고 바꿔주세요

    train_split_rate = "[0%:5%]"
    test_size = 0.2
    data_dir = "/content/drive/MyDrive/audio_classification/dataset/train"
CFG = config()


def load_dataset_from_directory(data_dir = CFG.data_dir, train_split_rate = CFG.train_split_rate, test_size = CFG.test_size):
    all_train_data = load_dataset("audiofolder",
                            name = "audio",
                            data_dir = data_dir,
                            split="train" + train_split_rate)
    all_train_data = all_train_data.train_test_split(test_size=0.2)
    all_train_data = all_train_data.cast_column("audio", Audio(sampling_rate=CFG.sampling_rate))
    return all_train_data

def get_prepare_dataset_func(feature_extractor):
    feature_extractor = feature_extractor
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        
        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=CFG.sampling_rate).input_features[0]

        # encode target text to label ids 
        batch["labels"] = batch["label"]
        return batch
    return prepare_dataset
    

def main():
    print("loading dataset from directory...")
    ds = load_dataset_from_directory()

    print("preprocessing dataset...")
    encoded_ds = ds.map(prepare_dataset)
    return encoded_ds
