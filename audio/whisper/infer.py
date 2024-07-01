import glob
import transformers
from transformers import pipeline

class config:
    MODEL = "/content/drive/MyDrive/audio/models/output_2/" # 모델 경로(processor, model 둘 다 저장되어 있어야 함)
    CHUNK_LENGTH_S = 20.1 # 오디오 파일 너무 길면 20.1초로 끊어서 처리함
    BATCH_SIZE = 4 # test batch size
    DATASET_PATH = '/content/drive/MyDrive/audio/dataset/test_mp3s/' # test할 audio파일 들어있는 경로
    max_count = 20 # N회 이상 반복되는 단어 모두 제거

CFG = config()

# 너무 많이 반복되는 단어 제거
def fix_repetition(text, max_count):
    uniq_word_counter = {}
    words = text.split()
    for word in text.split():
        if word not in uniq_word_counter:
            uniq_word_counter[word] = 1
        else:
            uniq_word_counter[word] += 1

    for word, count in uniq_word_counter.items():
        if count > max_count:
            words = [w for w in words if w != word]
    text = " ".join(words)
    return text

def inference(target_language, max_length, enable_beams, num_beams):
    # load file path
    files = list(glob.glob(CFG.DATASET_PATH + '/' + '*.wav'))
    files += list(glob.glob(CFG.DATASET_PATH + '/' + '*.mp3'))
    files.sort()
    
    # load model
    pipe = pipeline(task="automatic-speech-recognition",
                    model=CFG.MODEL,
                    tokenizer=CFG.MODEL,
                    chunk_length_s=CFG.CHUNK_LENGTH_S, device=0, batch_size=CFG.BATCH_SIZE)
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=target_language, task="transcribe")
    if enable_beams:
        texts = pipe(files, generate_kwargs={"max_length": max_length, "num_beams": num_beams})
    else:
        texts = pipe(files)
    
    # 너무 많이 반복되는 단어 제거 후 output return
    texts = [{'text':fix_repetition(d['text'], CFG.max_count)} for d in texts]
    return texts

def inference_with_dataloader(target_language, max_length, enable_beams, num_beams):
    pass