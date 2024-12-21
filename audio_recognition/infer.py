import glob
import argparse
import transformers
from transformers import pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration for model testing")
    # Define arguments corresponding to the config values
    parser.add_argument('model', type=str, required=True, help="Path to the model directory")
    parser.add_argument('dataset_path', type=str, required=True, help="Path to the test audio files directory")
    parser.add_argument('--chunk_length_s', type=float, default=20.1, help="Chunk length in seconds for audio splitting")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for testing")
    parser.add_argument('--max_count', type=int, default=20, help="Maximum count for repeating words to remove")

    # Parse the arguments
    args = parser.parse_args()
    return args


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


def inference(target_language, max_length, enable_beams, num_beams, model, dataset_path, chunk_length_s, batch_size, max_count):
    # load file path
    files = list(glob.glob(dataset_path + '/' + '*.wav'))
    files += list(glob.glob(dataset_path + '/' + '*.mp3'))
    files.sort()
    
    # load model
    pipe = pipeline(task="automatic-speech-recognition",
                    model=model,
                    tokenizer=model,
                    chunk_length_s=chunk_length_s, device=0, batch_size=batch_size)
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=target_language, task="transcribe")
    if enable_beams:
        texts = pipe(files, generate_kwargs={"max_length": max_length, "num_beams": num_beams})
    else:
        texts = pipe(files)
    
    # 너무 많이 반복되는 단어 제거 후 output return
    texts = [{'text':fix_repetition(d['text'], max_count)} for d in texts]
    return texts


def main():
    args = parse_arguments()
    inference('en', 128, True, 4, args.model, args.dataset_path, args.chunk_length_s, args.batch_size, args.max_count)
    

if __name__=="__main__":
    main()