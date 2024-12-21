import os
import yaml

def load_yaml(yml_path):
    # YAML 파일을 읽어서 Python 딕셔너리로 변환
    with open(yml_path, 'r') as yml_file:
        yml_data = yaml.safe_load(yml_file)
    return yml_data


def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f"Alert: Directory {directory} already exists -> pretrained model could be overwritten")