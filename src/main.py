from src.utils.common import read_yaml, create_directory
import argparse
import logging
import mlflow
import os

STAGE = "MAIN"

create_directory(["logs"])

with open(os.path.join("logs", 'running_logs.log'), "w") as f:
    f.write("")

logging.basicConfig(filename=os.path.join("logs", "running_logs.log"),
                    level=logging.INFO,
                    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
                    filemode="a")


def main():
    with mlflow.start_run() as run:
        mlflow.run(".", "stage_01_get_data", use_conda=False)
        mlflow.run(".", "stage_02_data_loader", use_conda=False)
        mlflow.run(".", "stage_03_model_creation", use_conda=False)
        mlflow.run(".", "stage_04_training_model", use_conda=False)
        mlflow.run(".", "stage_05_prediction", use_conda=False)


if __name__ == '__main__':
    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")

    except  Exception as e:
        logging.exception(e)
        raise e