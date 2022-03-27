
from torchvision import transforms, datasets
from src.stage_01_get_data import get_data
from src.utils.common import read_yaml
from torch.utils.data import DataLoader
import argparse
import logging
import torch
import os

STAGE = 'stage_02_data_loader'

log_file = os.path.join('logs', 'running_logs.log')

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format= "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode='a')

def data_loader(config_path):
    config = read_yaml(config_path)
    data = get_data(config_path)
    logging.info(f'Configuration File Loaded')

    train_data_loader = DataLoader(dataset = data[0], batch_size = config['Params']['BATCH_SIZE'], 
                                    shuffle = True)

    test_data_loader = DataLoader(dataset = data[1], batch_size = config['Params']['BATCH_SIZE'], 
                                    shuffle = False)
    logging.info(f'The Data Will Be Loaded Using Data Loader')

    return train_data_loader, test_data_loader, data[2]


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")

        data_loader(config_path=parsed_args.config)

        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e