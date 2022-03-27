from distutils.command.config import config
from src.utils.common import read_yaml, create_directory
from src.stage_02_data_loader import data_loader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import logging
import torch
import os

STAGE = "stage_03_model_creation"

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a")

class CNN(nn.Module):
    def __init__(self, in_ = 1, out_ = 10):
        super(CNN, self).__init__()
        logging.info(f'Creating The Base Model......')

        self.conv_pool_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=8, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_pool_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Flatten = nn.Flatten()
        self.FC_01 = nn.Linear(in_features=16*4*4, out_features=128)
        self.FC_02 = nn.Linear(in_features=128, out_features=64)
        self.FC_03 = nn.Linear(in_features=64, out_features=out_)
        logging.info(f'Base Model Created.....')

    def forward(self, x):
        logging.info(f'Making The Forword Pass......')

        x = self.conv_pool_1(x)
        x = self.conv_pool_2(x)
        x = self.Flatten(x)
        x = self.FC_01(x)
        x = F.relu(x)
        x = self.FC_02(x)
        x = F.relu(x)    
        x = self.FC_03(x)
        logging.info(f'Forword Pass Completed.....')

        return x


if __name__ == "__main__":
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")
        arg_parser = argparse.ArgumentParser()
        arg_parser.add_argument("--config", '-c', default="config/config.yaml")
        parsed_args = arg_parser.parse_args()

        config  = read_yaml(parsed_args.config)

        model_path = os.path.join(config['Artifacts']['MODEL'])

        create_directory([model_path])

        model_name = config['Artifacts']['BASE_MODEL']

        full_model_path = os.path.join(model_path, model_name)

        model_ob = CNN()

        torch.save(model_ob, full_model_path)

        logging.info(f"model created and saved at {full_model_path}")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        logging.exception(e)
        raise e