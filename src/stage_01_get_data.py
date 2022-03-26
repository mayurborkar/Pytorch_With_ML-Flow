from src.utils.common import read_yaml, create_directory
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import argparse
import logging
import torch
import os

STAGE = 'stage_01_get_data'

log_file = os.path.join('logs', 'running_logs.log')

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode='a')

def get_data(config_path): 
    config = read_yaml(config_path)
    data_folder_path = config['Data']['ROOT_DATA_FOLDER']
    create_directory([data_folder_path])
    logging.info(f'Directory Created Here {data_folder_path} For Getting Data')

    train_data = datasets.FashionMNIST(root = data_folder_path, train = True, download = True, 
                                        transform = transforms.ToTensor()
                                        )

    test_data = datasets.FashionMNIST(root = data_folder_path, train = False, download = True,
                                        transform = transforms.ToTensor()
                                        )
    logging.info(f'Both The Train & Test Data Download In {data_folder_path}')

    given_label = train_data.class_to_idx
    label_map = {val: key for key, val in given_label.items()}
    logging.info(f'Checking The Label Present In Data: {label_map}')

    return train_data, test_data, label_map


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")

        get_data(config_path=parsed_args.config)

        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e