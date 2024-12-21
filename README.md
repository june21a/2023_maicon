# 프로젝트 구조
```
📂EDA
 ┃ ┗ 📜audio_EDA.py
📂audio_classification
📦audio_recognition
 ┣ 📂whisper
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜utils.py
 ┃ ┣ 📜whisper_dataset.py
 ┃ ┣ 📜whisper_model.py
 ┃ 📜infer.py
 ┃ 📜module_test.ipynb
 ┃ 📜test_notebook.ipynb
 ┃ 📜torch_wav2vec_tutorial.ipynb
 ┃ 📜train_whisper.py
 ┣ 📂config
 ┣  ┗ 📜recognition_setting.yml
 ┣ 📜.gitignore
 ┣ 📜requirements.txt
```

# 사용방법
## installation
```
pip install -r requirements.txt
```

## train
- config의 recognition_setting.yml을 편집 후 다음과 같이 실행
  ```
  python train_whisper.py
  ```

## inference
```
python infer.py "/path/to/your/model" "/path/to/audio/files" --chunk_length_s 23.1 --batch_size 8 --max_count 15
```

# notice
- 대회가 audio recognition으로 진행되어서 audio classification 모듈을 실행시킬 수 있는데까지 만들고 버려진 상태(refactoring 필요)
