import logging
import yaml
import os

def read_yaml(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        content = yaml_file.safe_load()
    logging.info('Yaml File Loaded')
    return content

def create_directory(path_to_dir:list):
    """Create a directory If it doesn't exist'
    """
    full_path = ''
    for path in path_to_dir:
        full_path = os.path.join(full_path, path)
    os.mkdir(full_path, exist_ok=True)
    logging.info(f'Directory Created : {full_path}')