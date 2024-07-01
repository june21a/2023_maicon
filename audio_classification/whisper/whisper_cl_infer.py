import glob
import transformers
from transformers import pipeline

class config:
    MODEL = "/content/drive/MyDrive/audio/models/output_2/" # 모델 경로(processor, model 둘 다 저장되어 있어야 함)
    CHUNK_LENGTH_S = 20.1 # 오디오 파일 너무 길면 20.1초로 끊어서 처리함
    BATCH_SIZE = 4 # test batch size
    DATASET_PATH = '/content/drive/MyDrive/audio/dataset/test_mp3s/' # test할 audio파일 들어있는 경로