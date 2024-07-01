import os
import sys
import glob
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import concurrent.futures


import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from IPython.display import Audio

import ast
import librosa
from mutagen.mp3 import MP3
from pydub import AudioSegment

# check default metadata ( info, describe, head )
# 노트북 환경 아니면 display -> print 바꿔주기
def check_metadata(df):
    print("################# HEAD #################")
    display(df.head(10))
    print("\n\n")
    
    print("################# INFO #################")
    display(df.info())
    print("\n\n")
    
    print("################# DESC #################")
    display(df.describe())
    print("\n\n")


# get count of each unique value
def get_unique_values_count(df, col):
    cnts = []
    uniq = df[col].unique()
    for u in uniq:
        cnts.append(df[df[col] == u].count()[0])
    return cnts, uniq

# get ratio bar and array about column
def display_bar(df, columns):
    if len(columns) < 3:
        fig, axes = plt.subplots(1, 2)
        for i, col in enumerate(columns):
            cnts, uniq = get_unique_values_count(df, col)
            axes[i].bar(uniq, cnts)
    else:
        fig, axes = plt.subplpots(len(columns) // 2, 2)
        for i, col in enumerate(columns):
            cnts, uniq = get_unique_values_count(df, col)
            axes[i // 2][i % 2].bar(uniq, cnts)

# 히스토그램 그려주는 아이
def display_hist(df, columns):
    if len(columns) < 3:
        fig, axes = plt.subplots(1, 2)
        for i, col in enumerate(columns):
            sns.histplot(df[col].values, ax=axes[i])
    else:
        fig, axes = plt.subplpots(len(columns) // 2, 2)
        for i, col in enumerate(columns):
            sns.histplot(df[col].values, ax=axes[i // 2][i % 2])






# eda about total sentence ( 중복된 문장, 중복된 문장들 보여주기, 평균 길이 등)
def get_sentence_info(df, sentence_col):
    print("Total sentences :",len(df))
    print("Total unique sentences : ",df[sentence_col].nunique())
    print("Percentage of unique sentences ; ",df[sentence_col].nunique()/len(df))
    print("Most frequent sentences in the dataset \n")
    print(df[sentence_col].value_counts()[:10])
    df["len"] = df["sentence"].str.len()
    print(f"mean length of sentence : {df['len'].mean()}")

# 단어 단위로 분석한 결과를 보여줌
def get_all_vocab_count(df, sentence_col):
    vocab = {}
    for sen in tqdm(df[sentence_col]):
        for j in sen.split(" "):
            try:
                vocab[j]+=1
            except:
                vocab[j]=1
    print("Total words in vocabulary : ",len(vocab))
    sorted_vocab = sorted(vocab.items(),key = lambda kv:kv[1],reverse=True)
    print(f"vocab count from top :\n {sorted_vocab[:30]}")
    
    unique_words_count = [v for v in vocab.values()]
    # Compute descriptive statistics
    min_unique_words = min(unique_word_counts)
    max_unique_words = max(unique_word_counts)
    mean_unique_words = sum(unique_word_counts) / len(unique_word_counts)
    median_unique_words = sorted(unique_word_counts)[len(unique_word_counts) // 2]

    # Plot the distribution of unique word counts
    plt.figure(figsize=(10, 6))
    plt.hist(unique_word_counts, bins=50, color='lightcoral', edgecolor='black')
    plt.axvline(mean_unique_words, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_unique_words, color='green', linestyle='dashed', linewidth=2, label='Median')
    plt.xlabel('Number of Unique Words')
    plt.ylabel('Frequency')
    plt.title('Distribution of Unique Words in Transcriptions')
    plt.legend()
    plt.show()

    # Print the descriptive statistics
    print("Descriptive Statistics:")
    print(f"Minimum Number of Unique Words: {min_unique_words}")
    print(f"Maximum Number of Unique Words: {max_unique_words}")
    print(f"Mean Number of Unique Words: {mean_unique_words:.2f}")
    print(f"Median Number of Unique Words: {median_unique_words}")
    
    return vocab, sorted_vocab

# get_all_vocab_count에서 얻은 dict간 집합 연산을 수행하여 oov를 얻어냄
def get_oov_word_count(dict_1, dict_2):
    train_words = set([key for key,value in dict_1.items()])
    vocab_words = set([key for key,value in dict_2.items()])
    oov_set = vocab_words-train_words
    print("Total Out of vocabulary words : ",len(oov_set))
    return oov_set

# "문자" 단위로 분석한 결과를 보여줌
def get_char_info(df, sentence_col):
    chars = {}
    for sen in tqdm(df[sentence_col]):
        for j in sen:
            try:
                chars[j]+=1
            except:
                chars[j]=1
    print(f"all chars and count in the sentences :\n{chars}")
    print("\n\n")
    print("Total characters :",len(chars))
    print("\n\n\n")
    
    sorted_chars = sorted(chars.items(),key = lambda x:x[1],reverse = True)
    print(sorted_chars[:10])
    sorted_chars = dict(sorted_chars)
    plt.figure(figsize=(15,8))
    plt.title('Character Occurance')
    plt.bar(sorted_chars.keys(),sorted_chars.values())
    plt.xlabel('Unique characters')
    plt.ylabel('Sample Count')
    plt.grid(axis='y')
    plt.show()









# display the audio
def display_audio(audio_path):
    display(AudioSegment.from_file(audio_path))

# 경로 포함 file_list를 주면 랜덤한 오디오를 정답 label과 함께 보여줌
def display_random_audio_from_directory(file_list, df, id_col, sentence_col, count = 10):
#     file_list = glob.glob(directory + "/*")
    if len(file_list) < 3:
        indices = 1
        count = 1
    else:
        indices = np.random.randint(1, len(file_list)-1, count)
    for idx in indices:
        file_path = file_list[idx]
        file_id = Path(file_path).stem
        print(df[df[id_col] == file_id][sentence_col].values)
        display_audio(file_list[idx])
        
# mp3 로드하고 길이와 sampling rate를 얻어내는 친구
def process_mp3_file(mp3_file):
    try:
        audio = MP3(mp3_file)
        duration = audio.info.length
        sample_rate = audio.info.sample_rate
        return mp3_file, duration, sample_rate
    except Exception as e:
        print(f"Error processing {mp3_file}: {e}")
        return mp3_file, None, None

#배치단위로 mp3를 처리해줌
def process_mp3_files_in_batch(mp3_files):
    durs = []
    srs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_mp3_file, mp3_file) for mp3_file in mp3_files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            durs.append(result[1])
            srs.append(result[2])
    return durs,srs

# duration, sampling rate에 대한 정보를 알아냄
def durations(mp3_files_directory):
    # Get a list of all MP3 files in the directory
    mp3_files = glob.glob(os.path.join(mp3_files_directory, "*.mp3"))

    # Split the list of files into batches for processing
    batch_size = 1000
    file_batches = [mp3_files[i:i + batch_size] for i in tqdm(range(0, len(mp3_files), batch_size))]

    durs = []
    srs = []
    for batch in tqdm(file_batches):
        dur,sr = process_mp3_files_in_batch(batch)
        durs.extend(dur)
        srs.extend(sr)
    return durs,srs

def display_duration_hist(durs):
    plt.xlabel("Audio Length in seconds")
    plt.ylabel("Frequency")
    plt.title("Audio Length Distribuition")
    plt.hist(durs)

def display_sampling_rate(srs):
    print(f"sr이 동일한지 체크 : {set(srs)}")
    plt.xlabel("Sampling Rate")
    plt.ylabel("Frequency")
    plt.title("Sampling rate Distribuition")
    plt.hist(srs)

# draw wave, spectogram, chromagram, MFCCS
def display_audio_feature(audio_path):
   
    # Load an audio file
    samples, sample_rate = librosa.load(audio_path)

    # Visualize the waveform
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(samples, sr=sample_rate)
    plt.title('Waveform')

    # Compute the spectrogram
    spectrogram = librosa.stft(samples)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

    # Visualize the spectrogram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram (dB)')

    # Compute the mel spectrogram


    # Visualize the mel spectrogram
    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)

    # Visualize mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()


    # Compute the chromagram
    chromagram = librosa.feature.chroma_stft( y = samples , sr = sample_rate)

    # Visualize the chromagram
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(chromagram, sr=sample_rate, x_axis='time', y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')

    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)

    # Visualize the MFCCs
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    # Show the plots
    display(Audio(samples, rate=sample_rate))
    plt.show()