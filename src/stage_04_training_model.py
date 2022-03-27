from src.utils.common import read_yaml, create_directory
from src.stage_02_data_loader import data_loader
from src.stage_03_model_creation import CNN
from tqdm import tqdm 
import torch.nn as nn
import argparse
import logging
import torch
import os

STAGE = "stage_04_training_model"

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a")

def training_model(config_path): 
    try:
        config = read_yaml(config_path)
        train_data_loader, test_data_loader, label_map = data_loader(config_path)

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f'The Device Is Available In Machine Is {DEVICE} ')

        base_model_path = os.path.join(config['Artifacts']['MODEL'], config['Artifacts']['BASE_MODEL'])
        logging.info(f'The Base Model Path Is {base_model_path}')

        load_model = torch.load(base_model_path)
        load_model.eval()
        logging.info(f'Model Is Loaded')

        # load_model.to(DEVICE)
        # logging.info(f"{load_model} is loaded in {DEVICE}")  IF you Have GPU Uncomment These 2 Line

        learning_rate = config['Params']['LEARNING_RATE']
        epoch = config['Params']['EPOCH']
        criterion = nn.CrossEntropyLoss() ## loss function
        optimizer = torch.optim.Adam(load_model.parameters(), lr=learning_rate)
        logging.info(f'All The Parameters Are Loaded ')

        for epoch in range(epoch):
            with tqdm(train_data_loader) as tqdm_epoch:
                for images, labels in tqdm_epoch:
                    tqdm_epoch.set_description(f"Epoch {epoch + 1}/{epoch}") # 1st epoch is from for loop & 2nd is main epoch

                    # Uncomment Below 2 Lines If You Have GPU support
                    # images = images.to(config.DEVICE)
                    # labels = labels.to(config.DEVICE)

                    # forward pass
                    outputs = load_model(images)
                    loss = criterion(outputs, labels) # TODO #<< passing the pred, target

                    # backward prop
                    optimizer.zero_grad() # past gradient
                    loss.backward() # calculate the gradients
                    optimizer.step() # weights updated

                    tqdm_epoch.set_postfix(loss=loss.item())

        logging.info(f"Model trained successfully")
        trained_model_path = os.path.join(config['Artifacts']['MODEL'], config['Artifacts']['TRAINED_MODEL'])

        torch.save(load_model, trained_model_path)
        logging.info(f'trained model saved at {trained_model_path}')


    except Exception as e:
        logging.exception(e)
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")

        training_model(config_path=parsed_args.config)

        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e