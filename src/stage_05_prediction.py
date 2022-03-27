from src.utils.common import read_yaml, create_directory
from src.stage_02_data_loader import data_loader
import torch.nn.functional as F
import argparse
import logging
import torch
import os

STAGE = "stage_05_prediction"

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a")

def prediction(config_path):
    config = read_yaml(config_path)
    train_data_loader, test_data_loader, label_map = data_loader(config_path)

    test_data_batch = config['Params']['TEST_DATA_BATCH']

    pred_dir = os.path.join(config['Data']['PREDICTION_DATA'])
    create_directory([pred_dir])
    logging.info(f'The Dir Is Created Here {pred_dir})')

    pred_file = os.path.join(pred_dir, config['Data']['PREDICTION_FILE'])
    logging.info(f'The Prediction File Is Created Here {pred_file})')

    with open(pred_file, 'w') as f:
        f.write('')

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f'The Device Is Available In Machine Is {DEVICE} ')

    train_model_path = os.path.join(config['Artifacts']['MODEL'], config['Artifacts']['TRAINED_MODEL'])
    train_model = torch.load(train_model_path)
    logging.info(f'Model Is Loaded....')

    # model switched in cuda if present otherwise run it on cpu
    train_model.to(DEVICE)
    train_model.eval()
    logging.info(f'Model switched to Device {DEVICE}')

    c=0
    count=0
    for images, lables in test_data_loader:
        if c >= test_data_batch:
            break
        c = c + 1
        #load images in cuda If Present Otherwise run it on cpu
        images = images.to(DEVICE)
        lables = lables.to(DEVICE)
        logging.info(f'The Images & Label Run It On Device {DEVICE}')

        logit = train_model(images) # raw output
        actual = lables
        logging.info(f'The Output We Get After Applying Model On Images Is In The Raw Format: {logit}')

        for raw_output, label in zip(logit, actual):
            count +=1
            pred_prob = F.softmax(raw_output)
            argmax = torch.argmax(pred_prob).item()
            logging.info(f'The Argmax Value Is {argmax}')

            pred_class = label_map[argmax]
            logging.info(f'The Prediction class is {pred_class}')

            actual_class = label_map[label.item()]
            logging.info(f'The Actual class is {actual_class}')

            with open(pred_file, 'a+') as f:
                    f.write(f"Predicted class is---->{pred_class} Actual class is {actual_class}\n")

                    print(f"Predicted class is---->{pred_class} Actual class is {actual_class}")

    print(f"total count is {count}")
    logging.info(f"total count is {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parsed_args = parser.parse_args()
    try:
        logging.info("\n************************************")
        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} started<<<<<<<<<<<<<<<")

        prediction(config_path=parsed_args.config)

        logging.info(f">>>>>>>>>>>>>>>>>{STAGE} completed<<<<<<<<<<<<<<<")
        logging.info("\n************************************")
        
    except Exception as e:
        print(e)
        logging.exception(e)
        raise e